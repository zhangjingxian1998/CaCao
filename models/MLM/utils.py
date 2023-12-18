import json
import os
from statistics import median
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageFile
from transformers import ViTFeatureExtractor
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
coarse = ['above', 'across', 'against', 'along', 'and', 'at', 'behind', 'between', 'for', 'from', 'has', 'in', 'in front of', 'near', 'of', 'on', 'over', 'to', 'under', 'with']
fine = ['attached to', 'belonging to', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'growing on', 'hanging from', 'holding', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'on back of', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears']
gqa_info = json.load(open('/home/Datasets/GQA/GQA_200_ID_Info.json'))
gqa_predicate_set = gqa_info['ind_to_predicates'][1:]   

# vg-50 categories for stanford VG
vg_words = [line.strip('\n').strip('\r') for line in open('datasets/vg/predicate_list.txt')]
# total 587 categories for open-world VG
total_words = [line.strip('\n').strip('\r') for line in open('datasets/vg/extra_predicates_list.txt')]
# print(len(total_words))

def load_vg_dataset_image_text(path):
    all_triplets = []
    # without last 3 for 568, else for 565  
    # without up for 565, else for 562
    general_words = ['on', 'has', 'in', 'is', 'of', 'at', 'near', 'with', 'above', 'holding', 'behind', 'under', 'and', 'over', 'to', 'along', 'at', 'from', 'over', 'for', 'by', 'are', 'as', 'while', '\'s'] 
    image_info = json.load(open(path, 'r', encoding='UTF-8'))
    predicate_count_dict = dict()
    for k in image_info.keys():
        triplets = image_info[k]['triplets']
        for triplet in triplets:
            predicates = triplet[1].lower()
            if predicates in total_words:
                if predicates in predicate_count_dict.keys():
                    predicate_count_dict[predicates] += 1
                else:
                    predicate_count_dict[predicates] = 1
    # filter predicates only keep number is more than 10 common fine-grained predicates
    freq_fine_predicates = dict()
    temp_count = dict()
    for count_k in predicate_count_dict:
        if predicate_count_dict[count_k] > 3000:
            freq_fine_predicates[count_k] = 3000
            temp_count[count_k] = 3000
        elif predicate_count_dict[count_k] < 10:
            continue
        else:
            freq_fine_predicates[count_k] = predicate_count_dict[count_k]
            temp_count[count_k] = predicate_count_dict[count_k]
    for k in image_info.keys():
        triplets = image_info[k]['triplets']
        image_file = image_info[k]['image_file']
        for triplet in triplets:
            predicates = triplet[1].lower()
            if predicates in freq_fine_predicates and temp_count[predicates] > 0:                       
                image_triplet = dict()
                image_triplet['image_file'] = image_file
                image_triplet['triplet'] = triplet
                all_triplets.append(image_triplet)
                temp_count[predicates] -= 1
    predicates_preq = []
    predicates_words = []
    for key in total_words:
        if key in freq_fine_predicates.keys():
            predicates_words.append(key)
            predicates_preq.append(freq_fine_predicates[key])
        else:
            predicates_words.append(key)
            predicates_preq.append(10)
    # compute initial weight for adaptive loss  
    mid = median(predicates_preq)
    weight = []
    for c in predicates_preq:
        weight.append(mid/c)
    return all_triplets, weight, predicates_words
def load_vg_mapping_dataset_image_text(path):
    all_triplets = []
    image_info = json.load(open(path, 'r', encoding='UTF-8'))
    predicate_count_dict = dict()
    for k in image_info.keys():
        triplets = image_info[k]['triplets']
        for triplet in triplets:
            predicates = triplet[1].lower()
            if predicates in vg_words:                                                 ##在VG数据集中  计数+1
                if predicates in predicate_count_dict.keys():
                    predicate_count_dict[predicates] += 1
                else:
                    predicate_count_dict[predicates] = 1
    freq_fine_predicates = dict()
    m = 0
    for k in predicate_count_dict:
        m += predicate_count_dict[k]
    temp_count = dict()
    for count_k in predicate_count_dict:
        if predicate_count_dict[count_k] > 3000:
            freq_fine_predicates[count_k] = 3000
            temp_count[count_k] = 3000
        else:
            freq_fine_predicates[count_k] = predicate_count_dict[count_k]
            temp_count[count_k] = predicate_count_dict[count_k]                             ##得到freq_fine_predicates（出现次数高于3000的设为3000）
    for k in image_info.keys():
        triplets = image_info[k]['triplets']
        image_file = image_info[k]['image_file']
        for triplet in triplets:
            predicates = triplet[1].lower()
            if predicates in freq_fine_predicates and temp_count[predicates] > 0:                     
                image_triplet = dict()
                image_triplet['image_file'] = image_file
                image_triplet['triplet'] = triplet
                all_triplets.append(image_triplet)
                temp_count[predicates] -= 1                                                ##得到all_triplets（image_file + triplet）

    predicates_preq = []
    predicates_words = [] 
    for key in vg_words:
        if key in freq_fine_predicates.keys():
            predicates_words.append(key)
            predicates_preq.append(freq_fine_predicates[key])
        else:
            predicates_words.append(key)                    ##得到 predicates_words是VG 50类 predicates_preq是对应的数量表（没有出现过的谓词取10）
            predicates_preq.append(10)
    mid = median(predicates_preq)                           ##出现次数的中位数
    weight = []
    for c in predicates_preq:
        weight.append(mid/c)                                ##权重
    return all_triplets, weight, predicates_words

def load_gqa_dataset_image_text(self):
        all_triplets = []
        image_info = json.load(open(self.path, 'r', encoding='UTF-8'))
        predicate_count_dict = dict()
        # filter finetune for dataset
        for k in image_info:
            image_file = image_info[k]['image_file']
            triplets = image_info[k]['triplets']
            for triplet in triplets:
                predicates = triplet[1].lower()
                if predicates in gqa_predicate_set and predicates not in coarse:
                    if predicates in predicate_count_dict.keys() and predicate_count_dict[predicates] < 3000:
                        image_triplet = dict()
                        image_triplet['image_file'] = image_file
                        image_triplet['triplet'] = triplet
                        all_triplets.append(image_triplet)
                        predicate_count_dict[predicates] += 1
                    elif predicates in predicate_count_dict:
                        continue
                    else:
                        image_triplet = dict()
                        image_triplet['image_file'] = image_file
                        image_triplet['triplet'] = triplet
                        all_triplets.append(image_triplet)
                        predicate_count_dict[predicates] = 1

        predicates_preq = []
        predicates_words = []
        for key in gqa_predicate_set:
            if key not in coarse:
                if key in predicate_count_dict.keys() and predicate_count_dict[key] > 10:
                    predicates_words.append(key)
                    predicates_preq.append(predicate_count_dict[key])
                else:
                    predicates_words.append(key)
                    predicates_preq.append(10)
        mid = median(predicates_preq)
        weight = []
        for c in predicates_preq:
            weight.append(mid/c)
        return all_triplets, weight, predicates_words
class fineTuningDataset(Dataset):
    def __init__(self, img_root, all_triplets, weight, predicates_words, mode=None):
        # self.triplets, self.weight, self.predicates_words = self.load_vg_dataset_image_text()
        # self.triplets, self.weight, self.predicates_words = self.load_vg_mapping_dataset_image_text()
        # add extra vg-50 predicates data
        self.triplets=all_triplets
        self.weight=weight
        self.predicates_words=predicates_words
        self.img_root = img_root
        # np.random.shuffle(self.triplets)
        if mode == "train": # 80%
            self.triplets = self.triplets[:int(0.80*len(self.triplets))]
        elif mode == "test": # 10% = 80%~90%
            self.triplets = self.triplets[int(0.80*len(self.triplets)):int(0.9 * len(self.triplets))]
        else:               # 10% = 90%~100%
            self.triplets = self.triplets[int(0.90*len(self.triplets)):]
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('vit-base-patch32-224-in21k')
    
    
    def __len__(self):
        return len(self.triplets)
    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x
    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.triplet
        # img: 'COCO_train2014_000000455770.jpg'
        # triplet: '['hooks','hanging on','bathroom']'
        triplet = self.triplets[idx]

        img_info = self.triplets[idx]
        img_path = os.path.join(self.img_root,img_info['image_file'])
        triplet = img_info['triplet']

        # tf = transforms.Compose([
        #     lambda x:Image.open(x).convert("RGB"), # string path => image data
        #     transforms.Resize(224),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        image = Image.open(img_path).convert("RGB")
        img = self.feature_extractor(image)
        return img_path, '--'.join(triplet), img

def layer_init(layer, init_para=0.1, normal=False, xavier=True):
    xavier = False if normal == True else True
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
        torch.nn.init.constant_(layer.bias, 0)
        return
    elif xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0)
        return 