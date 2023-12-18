import torch
from torch import nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor
from models.MLM.models import BertForPromptFinetuning
from models.MLM.MultiHeadAttention import TransformerLayer
from models.MLM.modeling_vit import ViTModel

from models.MLM.modeling_bert import BertForMaskedLM
from models.MLM.tokenization_bert_fast import BertTokenizerFast
import time
from PIL import Image


class VisualBertPromptModel(nn.Module):
    def __init__(self, prefix_prompt_num, prompt_candidates, predicates_words, hidden_size=768, relation_type_count=31):
        super().__init__()
        self.prompt_num = prefix_prompt_num                                       #10
        self.prefix_prompt = prompt_candidates[:prefix_prompt_num]                ####前10个
        self.word_table = predicates_words
        prompt_ids = []
        for i in range(prefix_prompt_num):
            prompt_ids.append(i+1)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('vit-base-patch32-224-in21k')
        self.visual_encoder = ViTModel.from_pretrained('vit-base-patch32-224-in21k')
        self.transformerlayer = TransformerLayer(hidden_size=hidden_size, num_attention_heads=12, attention_probs_dropout_prob=0.1, intermediate_size=3072, hidden_dropout_prob=0.1, layer_norm_eps=1e-8)
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', prompt_ids)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.embedding_input = self.model.get_input_embeddings()
        self.predicate_embeddings = []
        self.predicate_embeddings_fast = []
        with torch.no_grad():
            word_sentence = ' [MASK] '.join(self.word_table) # 拼成句子，中间用 [MASK] 组合，接下来encode后，把MASK的地方作为分割标志
            predicate_int = self.tokenizer.encode(word_sentence, return_tensors="pt", add_special_tokens = False)
            predicate_embedding = self.embedding_input(predicate_int) # [1,N,768]
            index = torch.where(predicate_int==103)[-1]
            for i in range(index.shape[-1]):
                if i == 0:
                    self.predicate_embeddings_fast.append(torch.mean(predicate_embedding[0,0:index[i]],dim=0,keepdim=True))
                else:
                    self.predicate_embeddings_fast.append(torch.mean(predicate_embedding[0,index[i-1]+1:index[i]],dim=0,keepdim=True))
            self.predicate_embeddings_fast.append(torch.mean(predicate_embedding[0,index[-1]+1:],dim=0,keepdim=True))       
        self.prompt4re = BertForPromptFinetuning(self.model.config, relation_type_count, self.model.cls, self.word_table, self.predicate_embeddings_fast)
        self.max_input = 32
    def forward(self, batch_text, img_path, weight=None, theta=1.0, is_label=True):
        device = next(self.visual_encoder.parameters()).device
        image = Image.open(img_path).convert("RGB")
        image_input = self.feature_extractor(images=image, return_tensors="pt").to(device)
        
        text_input = torch.Tensor([]).to(device)
        label = []
        attention_mask = torch.Tensor([])
        mask_pos = []
        token_length = 30
        contexts = []
        #############################################
        for i in range(len(batch_text)):
            text_prompt = batch_text[i][0] + ' [MASK] ' + batch_text[i][2]                        ##'woman [MASK] boogie-board'
            contexts.append(' '.join([batch_text[i][0], batch_text[i][1], batch_text[i][2]]))     ##'woman laying on boogie-board'      
            label_id = [self.word_table.index(batch_text[i][1])] if is_label else [-1]            ##谓词ID eg:laying on--23
            # text_prompt = 'The relationship is '+text_prompt
            input = self.tokenizer.encode(text_prompt, return_tensors="pt", add_special_tokens = True)  
            token = []
            prompt_str = []
            for prompt in self.prefix_prompt:
                prompt_str.append(prompt)
            for prompt in prompt_str:
                token_id = self.tokenizer.convert_tokens_to_ids(prompt)
                token.append(token_id)
            token = torch.cat((torch.Tensor(token), input[0]), dim=0)         ##shape=10+len(input)
            token = torch.unsqueeze(token, dim=0)                             ##[1,10+len(input)]
            # The image is [CLS][Visual-prompt vectors] with [SUB] and [OBJ]. The relationship is [SUB][MASK][OBJ]
            # [CLS][Visual-prompt vectors] [CLS] The relationship is [SUB][MASK][OBJ] [SEP]
            # attention mask
            visual_len = 50
            token = add_zero(token, token_length)                              ##[1,30]
            mask = torch.ones_like(token) - make_mask(token)                   ##10+len(input)+补全  [1,30]
            visual_mask = torch.ones(1,visual_len)                             ##[1,50]  
            mask = torch.cat((visual_mask, mask), dim=1)                       ##[1,80]
            
            attention_mask = torch.cat((attention_mask, mask)).long()          ##[1,80]
            # labels
            mask_idx = []
            for m in range(len(token[0])):
                if token[0][m] == 103:
                    mask_idx.append(m + visual_len)                            ##12+50
            mask_pos.append(mask_idx)
            token = token.to(device)
            text_input = torch.cat((text_input, token)).long().to(device)      ##[1,30]
            label.append(label_id)
        image_in = {'pixel_values':image_input['pixel_values'],'output_attentions':True}
        outputs = self.visual_encoder(**image_in)        ##attention 12  last_hidden_state [1,50,768]  pooler_output [1,768]
        cls_feature = outputs.pooler_output                 ##[1,768]
        prompt4visual = torch.cat((torch.unsqueeze(cls_feature, dim=1), outputs.last_hidden_state[:,1:,:]), dim=1).to(device)     #unsqueeze指定维度上添加一个尺寸为1的维度
        visual_prompt = self.transformerlayer(prompt4visual)   #[1,50,768]
        label = torch.Tensor(label).long().to(device)                     ##[64,1]
        attention_mask = attention_mask.to(device)

        range_time = text_input.shape[0] // self.max_input
        rest = text_input.shape[0] - range_time * self.max_input
        final_torch = torch.tensor([]).to(device)
        if range_time > 0:
            for i in range(range_time):
                output = self.model(input_ids=text_input[self.max_input*i:self.max_input*(i+1)], 
                                    labels=label[self.max_input*i:self.max_input*(i+1)], 
                                    visual_prompts=visual_prompt.repeat(self.max_input,1,1), 
                                    attention_mask=attention_mask[self.max_input*i:self.max_input*(i+1)])   ##Bertmodel    只使用hidden_states作后续处理          
                mask_pos_tmp = torch.Tensor(mask_pos[self.max_input*i:self.max_input*(i+1)]).long()                          ##[64,1]
            
                weight = torch.Tensor(weight).to(device) if weight is not None else None
                final = self.prompt4re(output[1], mask_pos=mask_pos_tmp, labels=None, weight=weight, theta=theta, device=device)      #####loss+logits
                final_torch = torch.cat([final_torch,final[0]],dim=0)
        if rest > 0:
            output = self.model(input_ids=text_input[-rest:], 
                                        labels=label[-rest:], 
                                        visual_prompts=visual_prompt.repeat(rest,1,1), 
                                        attention_mask=attention_mask[-rest:])   ##Bertmodel    只使用hidden_states作后续处理          
            mask_pos_tmp = torch.Tensor(mask_pos[-rest:]).long()                          ##[64,1]
        
            weight = torch.Tensor(weight).to(device) if weight is not None else None
            final = self.prompt4re(output[1], mask_pos=mask_pos_tmp, labels=None, weight=weight, theta=theta, device=device)      #####loss+logits
            final_torch = torch.cat([final_torch,final[0]],dim=0)
        # logits = output.logits
        return final_torch
    def mapping_target(self, predicted_rel, target_words, prep_words, device='cuda:0'):
        embedding_model = self.model.bert.get_input_embeddings()
        if predicted_rel not in prep_words:
            predicted_keyword_split = predicted_rel.split(' ')
            for prep in prep_words:
                if prep in predicted_keyword_split:
                    predicted_keyword_split.remove(prep)
            predicted_rel = ' '.join(predicted_keyword_split)
        text_predict_prompt = self.tokenizer.encode(predicted_rel, return_tensors="pt", add_special_tokens = False).to(device)     #[1,1]
        predicted_embedding_token = embedding_model(text_predict_prompt)                        ##[1,1,768]
        predicted_embedding = torch.sum(predicted_embedding_token, dim=1)                       ##[1,768]
        predicted_embedding /= predicted_embedding_token.shape[1]                               ##得到当前谓词的embedding
        all_keywords_w2v_list = []
        for target_rel in target_words:
            with torch.no_grad():
                if target_rel not in prep_words:
                    original_target_rel = target_rel
                    keyword_split = target_rel.split(' ')
                    for prep in prep_words:
                        if prep in keyword_split:
                            keyword_split.remove(prep)
                    target_rel = ' '.join(keyword_split)
                # additional hierarchy structure for mapping
                text_target_prompt = self.tokenizer.encode(target_rel, return_tensors="pt", add_special_tokens = False).to(device)
                embedding_token = embedding_model(text_target_prompt)
                total_embedding = torch.sum(embedding_token, dim=1)
                total_embedding /= embedding_token.shape[1]
                all_keywords_w2v_list.append((original_target_rel, total_embedding))                ##50个谓词的embedding
        target_worddic = dict(all_keywords_w2v_list)

        similarity = [torch.cosine_similarity(predicted_embedding, target_worddic[w[0]], dim=1) for w in all_keywords_w2v_list]
        similarity = torch.cat(similarity, dim=-1)
        top_score = torch.topk(similarity, k=3)             ##找到得分最高的三个
        top_score_index = top_score[1]
        # max_score = max(similarity)
        # max_score_index = similarity.index(max_score)
        result_words = []
        for i in top_score_index:
            result_words.append(all_keywords_w2v_list[i][0])
        return result_words
            
def add_zero(token, token_length):
    zero = torch.Tensor([[0] * (token_length-token.shape[1])])
    return torch.cat((token, zero), dim=1)
def make_mask(feature):
    return (feature[0] == 0).long().unsqueeze(0)

    
