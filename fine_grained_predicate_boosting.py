# coding=utf8
# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

import argparse, json, string
import os
import random
from collections import Counter
import math
from nltk.corpus import wordnet as wn
from math import floor
import h5py as h5
import numpy as np

import torch
from tqdm import tqdm
from models.MLM.tokenization_bert_fast import BertTokenizerFast
from models.MLM.mpt_test_boost import VisualBertPromptModel
import time
from models.MLM.utils import load_vg_dataset_image_text, load_vg_mapping_dataset_image_text

"""
A script for generating fine-grained predicates for the VisualGenome dataset
"""

def preprocess_object_labels(data, alias_dict={}):
    for img in data:
        for obj in img['objects']:
            obj['ids'] = [obj['object_id']]
            names = []
            for name in obj['names']:
                label = sentence_preprocess(name)
                if label in alias_dict:
                    label = alias_dict[label]
                names.append(label)
            obj['names'] = names


def preprocess_predicates(data, alias_dict={}):
    for img in data:
        for relation in img['relationships']:
            predicate = sentence_preprocess(relation['predicate'])
            if predicate in alias_dict:
                predicate = alias_dict[predicate]
            relation['predicate'] = predicate


def extract_object_token(data, num_tokens, obj_list=[], verbose=True):
    """ Builds a set that contains the object names. Filters infrequent tokens. """
    token_counter = Counter()
    for img in data:
        for region in img['objects']:
            for name in region['names']:
                if not obj_list or name in obj_list:
                    token_counter.update([name])
    tokens = set()
    # pick top N tokens
    token_counter_return = {}
    for token, count in token_counter.most_common():
        tokens.add(token)
        token_counter_return[token] = count
        if len(tokens) == num_tokens:
            break
    if verbose:
        print(('Keeping %d / %d objects'
                  % (len(tokens), len(token_counter))))
    return tokens, token_counter_return


def extract_predicate_token(data, num_tokens, pred_list=[], verbose=True):
    """ Builds a set that contains the relationship predicates. Filters infrequent tokens. """
    token_counter = Counter()
    total = 0
    for img in data:
        for relation in img['relationships']:
            predicate = relation['predicate']
            if not pred_list or predicate in pred_list:
                token_counter.update([predicate])
            total += 1
    tokens = set()
    token_counter_return = {}
    for token, count in token_counter.most_common():
        tokens.add(token)
        token_counter_return[token] = count
        if len(tokens) == num_tokens:
            break
    if verbose:
        print(('Keeping %d / %d predicates with enough instances'
                  % (len(tokens), len(token_counter))))
    return tokens, token_counter_return

def IoU(b1, b2):
    if b1[2] <= b2[0] or \
        b1[3] <= b2[1] or \
        b1[0] >= b2[2] or \
        b1[1] >= b2[3]:
        return 0

    b1b2 = np.vstack([b1,b2])
    minc = np.min(b1b2, 0)
    maxc = np.max(b1b2, 0)
    union_area = (maxc[2]-minc[0])*(maxc[3]-minc[1])
    int_area = (minc[2]-maxc[0])*(minc[3]-maxc[1])
    return float(int_area)/float(union_area)

def to_x1y1x2y2(obj):
    x1 = obj['x']
    y1 = obj['y']
    x2 = obj['x'] + obj['w']
    y2 = obj['y'] + obj['h']
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def merge_duplicate_boxes(data):
    

    def inside(b1, b2):
        return b1[0] >= b2[0] and b1[1] >= b2[1] \
            and b1[2] <= b2[2] and b1[3] <= b2[3]

    def overlap(obj1, obj2):
        b1 = to_x1y1x2y2(obj1)
        b2 = to_x1y1x2y2(obj2)
        iou = IoU(b1, b2)
        # consider different type of overlapping
        if all(b1 == b2) or iou > 0.9: # consider as the same box
            return 1
        elif (inside(b1, b2) or inside(b2, b1))\
            and obj1['names'][0] == obj2['names'][0]: # same object inside the other
            return 2
        elif iou > 0.6 and obj1['names'][0] == obj2['names'][0]: # multiple overlapping same object
            return 3
        else:
            return 0  # no overlap

    num_merged = {1:0, 2:0, 3:0}
    print('merging boxes..')
    for img in data:
        # mark objects to be merged and save their ids
        objs = img['objects']
        num_obj = len(objs)
        for i in range(num_obj):
            if 'M_TYPE' in objs[i]:  # has been merged
                continue
            merged_objs = [] # circular refs, but fine
            for j in range(i+1, num_obj):
                if 'M_TYPE' in objs[j]:  # has been merged
                    continue
                overlap_type = overlap(objs[i], objs[j])
                if overlap_type > 0:
                    objs[j]['M_TYPE'] = overlap_type
                    merged_objs.append(objs[j])
            objs[i]['mobjs'] = merged_objs

        # merge boxes
        filtered_objs = []
        merged_num_obj = 0
        for obj in objs:
            if 'M_TYPE' not in obj:
                ids = [obj['object_id']]
                dims = [to_x1y1x2y2(obj)]
                prominent_type = 1
                for mo in obj['mobjs']:
                    ids.append(mo['object_id'])
                    obj['names'].extend(mo['names'])
                    dims.append(to_x1y1x2y2(mo))
                    if mo['M_TYPE'] > prominent_type:
                        prominent_type = mo['M_TYPE']
                merged_num_obj += len(ids)
                obj['ids'] = ids
                mdims = np.zeros(4)
                if prominent_type > 1: # use extreme
                    mdims[:2] = np.min(np.vstack(dims)[:,:2], 0)
                    mdims[2:] = np.max(np.vstack(dims)[:,2:], 0)
                else:  # use mean
                    mdims = np.mean(np.vstack(dims), 0)
                obj['x'] = int(mdims[0])
                obj['y'] = int(mdims[1])
                obj['w'] = int(mdims[2] - mdims[0])
                obj['h'] = int(mdims[3] - mdims[1])

                num_merged[prominent_type] += len(obj['mobjs'])

                obj['mobjs'] = None
                obj['names'] = list(set(obj['names']))  # remove duplicates

                filtered_objs.append(obj)
            else:
                assert 'mobjs' not in obj

        img['objects'] = filtered_objs
        assert(merged_num_obj == num_obj)

    print('# merged boxes per merging type:')
    print(num_merged)


def build_token_dict(vocab):
    """ build bi-directional mapping between index and token"""
    token_to_idx, idx_to_token = {}, {}
    next_idx = 1
    vocab_sorted = sorted(list(vocab)) # make sure it's the same order everytime
    for token in vocab_sorted:
        token_to_idx[token] = next_idx
        idx_to_token[next_idx] = token
        next_idx = next_idx + 1

    return token_to_idx, idx_to_token


def encode_box(region, org_h, org_w, im_long_size):
    x = region['x']
    y = region['y']
    w = region['w']
    h = region['h']
    scale = float(im_long_size) / max(org_h, org_w)
    image_size = im_long_size
    # recall: x,y are 1-indexed
    x, y = math.floor(scale*(region['x']-1)), math.floor(scale*(region['y']-1))
    w, h = math.ceil(scale*region['w']), math.ceil(scale*region['h'])

    # clamp to image
    if x < 0: x = 0
    if y < 0: y = 0

    # box should be at least 2 by 2
    if x > image_size - 2:
        x = image_size - 2
    if y > image_size - 2:
        y = image_size - 2
    if x + w >= image_size:
        w = image_size - x
    if y + h >= image_size:
        h = image_size - y

    # also convert to center-coord oriented
    box = np.asarray([x+floor(w/2), y+floor(h/2), w, h], dtype=np.int32)
    assert box[2] > 0  # width height should be positive numbers
    assert box[3] > 0
    return box


def encode_objects(obj_data, token_to_idx, token_counter, org_h, org_w, im_long_sizes):
    encoded_labels = []
    encoded_boxes  = {}
    for size in im_long_sizes:
        encoded_boxes[size] = []
    im_to_first_obj = np.zeros(len(obj_data), dtype=np.int32)
    im_to_last_obj = np.zeros(len(obj_data), dtype=np.int32)
    obj_counter = 0

    for i, img in enumerate(obj_data):
        im_to_first_obj[i] = obj_counter
        img['id_to_idx'] = {}  # object id to region idx
        # obj_len = len(img['objects'])
        for obj in img['objects']:
           # pick a label for the object
            max_occur = 0
            obj_label = None
            for name in obj['names']:
                # pick the name that has maximum occurance
                if name in token_to_idx and token_counter[name] > max_occur:
                    obj_label = name
                    max_occur = token_counter[obj_label]

            if obj_label is not None:
                # encode region
                for size in im_long_sizes:
                    encoded_boxes[size].append(encode_box(obj, org_h[i], org_w[i], size))

                encoded_labels.append(token_to_idx[obj_label])

                for obj_id in obj['ids']: # assign same index for merged ids
                    img['id_to_idx'][obj_id] = obj_counter

                obj_counter += 1
        if im_to_first_obj[i] == obj_counter:
            im_to_first_obj[i] = -1
            im_to_last_obj[i] = -1
        else:
            im_to_last_obj[i] = obj_counter - 1

    for k, boxes in encoded_boxes.items():
        encoded_boxes[k] = np.vstack(boxes)
    return np.vstack(encoded_labels), encoded_boxes, im_to_first_obj, im_to_last_obj


def encode_relationship(sub_id, obj_id, id_to_idx):
    # builds a tuple of the index of object and subject in the object list
    sub_idx = id_to_idx[sub_id]
    obj_idx = id_to_idx[obj_id]
    return np.asarray([sub_idx, obj_idx], dtype=np.int32)
def predict_rel_prompt(model, img_path, subject, object):
    batch_text = []
    predicate = '[MASK]'
    for i in range(len(subject)):
        batch_text.append((subject[i].lower(), predicate, object[i].lower())) 
    with torch.no_grad():
        output = model(batch_text, img_path, is_label=False)
    predictions = output
    expand_results = [] 
    expand_results_mapping = []
    # candidate predicates prediction
    # word_candidates = torch.argsort(predictions[0], descending=True)[:1].tolist()
    word_candidates = torch.argmax(predictions,dim=-1)
    for i in range(word_candidates.shape[0]):
        expand_results.append((batch_text[i][0], model.word_table[word_candidates[i]], batch_text[i][2]))
    # if len(word_candidates) == 1:
    #     word_candidate = model.word_table[word_candidates[0]] 
    # else:
    #     # increase diversity of expanded words (without mapping) 
    #     if random.random() > 0.5:
    #         word_candidate = model.word_table[word_candidates[0]]
    #     else:
    #         i = random.randint(1, len(word_candidates)-1)
    #         word_candidate = model.word_table[word_candidates[i]]
    # expand_results.append((batch_text[0][0], word_candidate, batch_text[0][2])) 
    # target predicates mapping with structral information
    for j in range(word_candidates.shape[0]):
        word = model.word_table[word_candidates[j]]
        mapping_words = mapping_dict[word]
        if len(mapping_words) > 1:
            if random.random() > 0.2:
                mapping_word = mapping_words[0]
            else:
                i = random.randint(1, len(mapping_words)-1)
                mapping_word = mapping_words[i]
        else:
            mapping_word = mapping_words[0]
        expand_results_mapping.append((batch_text[j][0], mapping_word, batch_text[j][2]))
    # mapping_words = mapping_dict[word_candidate]
    # if len(mapping_words) > 1:
    #     if random.random() > 0.2:
    #         mapping_word = mapping_words[0]
    #     else:
    #         i = random.randint(1, len(mapping_words)-1)
    #         mapping_word = mapping_words[i]
    # else:
    #     mapping_word = mapping_words[0]
    # expand_results.append((batch_text[0][0], mapping_word, batch_text[0][2])) 
    return expand_results, expand_results_mapping
def predict_rel(model, subject, object, length, device):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    batch = []
    predicates = []
    for i in range(length):
        predicates.append('[MASK]')
    predicate = " ".join(predicates)
    batch.append((subject, predicate, object))  
    output, label = model(batch, device)
    predictions = output.logits
    word = []
        
    l = 0
    for i in range(len(label[0])):
        if label[0][i] != -100:
            word_candidates = torch.argsort(predictions[0, i], descending=True)[:1].tolist()
            word_candidates = tokenizer.convert_ids_to_tokens(word_candidates)
            word.append(word_candidates)
            l += 1
    assert l==length
    word_cat = []
    for i in range(len(word[0])):
        s = []
        for j in range(l):
            s.append(word[j][i])
        word_cat.append(" ".join(s))
    expand_results = []
    for i, predicate in enumerate(word_cat):
        expand_results.append((batch[0][0],predicate,batch[0][2]))
    return expand_results
def expand_relationships(rel_data, obj_data, img_data, split, encoded_label, idx_to_label, im_to_first_obj, im_to_last_obj, encoded_boxes, expand_predicate_to_idx):
    prompt_candidates = []
    with open('bert-base-uncased/prompt.txt','r') as f:
        for line in f.readlines():
            prompt_candidates.append(line.strip('\n'))
    prompt_num = 10
    model_own = VisualBertPromptModel(prompt_num, prompt_candidates, words, relation_type_count=len(words))
    model_own.load_state_dict(torch.load('checkpoints/cluster_50_model.pkl'))
    model_own = model_own.to(device)
    n = 0 
    max_rel_id = 4836654
    new_predicate_dict = {}
    expand_dataset = dict()
    rst = []
    expand_relation_dict = dict()
    for i, rel_info in tqdm(enumerate(rel_data)):
        if i > 20:
            break
        if split[i] == 0:
            obj_info = obj_data[i]
            inter_objects = []
            assert obj_info['image_id'] == rel_info['image_id']
            id_to_idx = obj_data[i]['id_to_idx']
            if len(id_to_idx) == 0: # 50
                continue
            for obj1 in obj_info['objects']:
                for obj2 in obj_info['objects']:
                    b1 = to_x1y1x2y2(obj1)
                    b2 = to_x1y1x2y2(obj2)
                    iou = IoU(b1, b2)
                    # inter_objects.append((obj1,obj2))
                    if iou>0 and iou<1.0:
                        inter_objects.append((obj1,obj2))
            img_id = rel_info['image_id']
            rels = rel_info['relationships']
            
            new_rels = []
            original_objects = []
            expand_dataset[str(img_id)] = list()
            for rel in rels:
                object = rel['object']
                subject = rel['subject']
                if subject['object_id'] in id_to_idx and object['object_id'] in id_to_idx:
                    original_objects.append((id_to_idx[subject['object_id']], id_to_idx[object['object_id']]))
            basename =  str(img_id) + '.jpg'
            img_path = os.path.join("datasets/vg/VG_100K", basename)

            # construct new_rel_dict for extra.pk
            new_rel_dict =  {}
            new_rel_dict['image_id'] = i
            new_rel_dict['width'] = img_data[i]['width']
            new_rel_dict['height'] = img_data[i]['height']
            new_rel_dict['img_path'] = img_path
            scale = 1024
            all_boxes = encoded_boxes[scale]
            all_labels = encoded_label
            new_rel_dict['boxes'] = np.array(all_boxes[im_to_first_obj[i] : im_to_last_obj[i] + 1, :] / scale * max(img_data[i]['width'], img_data[i]['height']))
            new_rel_dict['labels'] = np.array(all_labels[im_to_first_obj[i] : im_to_last_obj[i] + 1]).reshape((1,-1))
            new_rel_dict['relations'] = []
            expand_relation = []

            subject_label_list = []
            object_label_list = []
            inter_object_list = []
            for inter_object in inter_objects:
                subject = inter_object[0]
                object = inter_object[1]
                if subject['object_id'] in id_to_idx and object['object_id'] in id_to_idx:
                    if (id_to_idx[subject['object_id']], id_to_idx[object['object_id']]) not in original_objects:
                        subject_label = idx_to_label[encoded_label[id_to_idx[subject['object_id']]][0]]
                        object_label = idx_to_label[encoded_label[id_to_idx[object['object_id']]][0]]
                        if subject_label != object_label:
                            subject_label_list.append(subject_label)
                            object_label_list.append(object_label)
                            inter_object_list.append(inter_object)
            if len(subject_label_list) == 0: # 66
                continue
            predicate_results, predicate_results_mapping = predict_rel_prompt(model_own, img_path, subject_label_list, object_label_list)
            expand_relation.extend(predicate_results)
            for idx, relationship in enumerate(predicate_results):
                subject = inter_object_list[idx][0]
                object = inter_object_list[idx][1]
                new_relation = {}
                new_predicate = relationship[1]
                new_object = object
                new_subject = subject
                max_rel_id += 1
                new_relationship_id = max_rel_id
                # new_synsets = wn.synsets(new_predicate)[0]
                new_synsets = [new_predicate + '.n.01']
                new_relation['predicate'] = new_predicate
                if new_predicate not in new_predicate_dict.keys():
                    new_predicate_dict[new_predicate] = 1
                else:
                    new_predicate_dict[new_predicate] += 1 
                new_relation['object'] = new_object
                new_relation['relationship_id'] = new_relationship_id
                new_relation['synsets'] = new_synsets
                new_relation['subject'] = new_subject
                new_rels.append(new_relation)
                
                sub_idx = id_to_idx[subject['object_id']] - im_to_first_obj[i]
                obj_idx = id_to_idx[object['object_id']] - im_to_first_obj[i]
                rel_id = expand_predicate_to_idx[relationship[1]]
                new_rel_dict['relations'].append([sub_idx, obj_idx, rel_id])
                expand_dataset[str(img_id)].append(relationship)


                new_relation = {}
                new_predicate = predicate_results_mapping[idx][1]
                # new_object = object_label_list[idx]
                # new_subject = subject_label_list[idx]
                max_rel_id += 1
                new_relationship_id = max_rel_id
                # new_synsets = wn.synsets(new_predicate)[0]
                new_synsets = [new_predicate + '.n.01']
                new_relation['predicate'] = new_predicate
                if new_predicate not in new_predicate_dict.keys():
                    new_predicate_dict[new_predicate] = 1
                else:
                    new_predicate_dict[new_predicate] += 1 
                new_relation['object'] = new_object
                new_relation['relationship_id'] = new_relationship_id
                new_relation['synsets'] = new_synsets
                new_relation['subject'] = new_subject
                new_rels.append(new_relation)
                
                sub_idx = id_to_idx[subject['object_id']] - im_to_first_obj[i]
                obj_idx = id_to_idx[object['object_id']] - im_to_first_obj[i]
                rel_id = expand_predicate_to_idx[relationship[1]]
                new_rel_dict['relations'].append([sub_idx, obj_idx, rel_id])
                expand_dataset[str(img_id)].append(relationship)

            new_rel_dict['relations'] = np.array(new_rel_dict['relations']) # expanded relationships like IETrans(external) with .pk
            expand_relation_dict[img_id] = expand_relation # expanded information for CaCao
            rst.append(new_rel_dict)
            # expand original dataset with CaCao fine-grained mapping
            rel_info['relationships'].extend(new_rels)
            # only use the expanded dataset without original data(unsupervised)
            # rel_info['relationships'] = new_rels
            n += len(new_rels)
    # expand_data_information: ['width', 'height', 'img_path', 'boxes', 'labels', 'relations'], save external relationship to pk
    return n, new_predicate_dict
def relationships_info(rel_data, token_to_idx, obj_data, predicate_token_counter):
    encoded_pred = []  # encoded predicates
    encoded_rel = []  # encoded relationship tuple
    im_to_first_rel = np.zeros(len(rel_data), dtype=np.int32)
    im_to_last_rel = np.zeros(len(rel_data), dtype=np.int32)
    rel_idx_counter = 0
    filter_predicate_counter = predicate_token_counter.copy()
    no_rel_counter = 0
    obj_filtered = 0
    predicate_filtered = 0
    duplicate_filtered = 0

    for i, img in enumerate(rel_data):
        im_to_first_rel[i] = rel_idx_counter
        id_to_idx = obj_data[i]['id_to_idx']  # object id to object list idx
        # rel_len = len(img['relationships'])
        for relation in img['relationships']:
            subj = relation['subject']
            obj = relation['object']
            predicate = relation['predicate']
            if subj['object_id'] not in id_to_idx or obj['object_id'] not in id_to_idx:
                obj_filtered += 1
                if predicate in filter_predicate_counter:
                    filter_predicate_counter[predicate] -= 1
            elif predicate not in token_to_idx:
                predicate_filtered += 1
            elif id_to_idx[subj['object_id']] == id_to_idx[obj['object_id']]: # sub and obj can't be the same box
                duplicate_filtered += 1
                if predicate in filter_predicate_counter:
                    filter_predicate_counter[predicate] -= 1
            else:
                encoded_pred.append(token_to_idx[predicate])
                encoded_rel.append(
                    encode_relationship(subj['object_id'],
                                        obj['object_id'],
                                        id_to_idx
                                        ))
                rel_idx_counter += 1  # accumulate counter

        if im_to_first_rel[i] == rel_idx_counter:
            # if no qualifying relationship
            im_to_first_rel[i] = -1
            im_to_last_rel[i] = -1
            no_rel_counter += 1
        else:
            im_to_last_rel[i] = rel_idx_counter - 1
    print('%i rel is filtered by object' % obj_filtered)
    print('%i rel is filtered by predicate' % predicate_filtered)
    print('%i rel is filtered by duplicate' % duplicate_filtered)
    print('%i rel remains ' % len(encoded_pred))
    print('%i out of %i valid images have relationships' % (len(rel_data)-no_rel_counter, len(rel_data)))
    return np.vstack(encoded_pred), np.vstack(encoded_rel), im_to_first_rel, im_to_last_rel, filter_predicate_counter
     
def encode_relationships(rel_data, token_to_idx, obj_data, new_predicate_dict, predicate_token_counter):
    """MUST BE CALLED AFTER encode_objects!!!"""
    # print(obj_data[0]['id_to_idx'])
    encoded_pred = []  # encoded predicates
    encoded_rel = []  # encoded relationship tuple
    im_to_first_rel = np.zeros(len(rel_data), dtype=np.int32)
    im_to_last_rel = np.zeros(len(rel_data), dtype=np.int32)
    rel_idx_counter = 0
    filter_predicate_counter = predicate_token_counter.copy()
    no_rel_counter = 0
    obj_filtered = 0
    predicate_filtered = 0
    duplicate_filtered = 0
    remain_expand_predicate = 0

    for k in new_predicate_dict.keys():
        if k in token_to_idx:
            remain_expand_predicate += new_predicate_dict[k]

    for i, img in enumerate(rel_data):
        im_to_first_rel[i] = rel_idx_counter
        id_to_idx = obj_data[i]['id_to_idx']  # object id to object list idx
        # print(len(img['relationships']))
        for relation in img['relationships']:
            subj = relation['subject']
            obj = relation['object']
            predicate = relation['predicate']
            if subj['object_id'] not in id_to_idx or obj['object_id'] not in id_to_idx:
                obj_filtered += 1
                if predicate in filter_predicate_counter:
                    filter_predicate_counter[predicate] -= 1
            elif predicate not in token_to_idx:
                predicate_filtered += 1
                if predicate in filter_predicate_counter:
                    filter_predicate_counter[predicate] -= 1
            elif id_to_idx[subj['object_id']] == id_to_idx[obj['object_id']]: # sub and obj can't be the same box
                duplicate_filtered += 1
                if predicate in filter_predicate_counter:
                    filter_predicate_counter[predicate] -= 1
            else:
                encoded_pred.append(token_to_idx[predicate])
                encoded_rel.append(
                    encode_relationship(subj['object_id'],
                                        obj['object_id'],
                                        id_to_idx
                                        ))
                rel_idx_counter += 1  # accumulate counter

        if im_to_first_rel[i] == rel_idx_counter:
            # if no qualifying relationship
            im_to_first_rel[i] = -1
            im_to_last_rel[i] = -1
            no_rel_counter += 1
        else:
            im_to_last_rel[i] = rel_idx_counter - 1
    print('%i rel is filtered by object' % obj_filtered)
    print('%i rel is filtered by predicate' % predicate_filtered)
    print('%i rel is filtered by duplicate' % duplicate_filtered)
    print('%i rel remains ' % len(encoded_pred))
    print('%i expanded rel is remained' % remain_expand_predicate)

    print('%i out of %i valid images have relationships' % (len(rel_data)-no_rel_counter, len(rel_data)))
    return np.vstack(encoded_pred), np.vstack(encoded_rel), im_to_first_rel, im_to_last_rel, filter_predicate_counter


def sentence_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    replacements = {
      '½': 'half',
      '—' : '-',
      '™': '',
      '¢': 'cent',
      'ç': 'c',
      'û': 'u',
      'é': 'e',
      '°': ' degree',
      'è': 'e',
      '…': '',
    }
    # phrase = phrase.encode('utf-8')
    phrase = phrase.lstrip(' ').rstrip(' ')
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(string.punctuation)
def encode_splits_random(obj_data, rel_data, data_split, opt=None):
    if opt is not None:
        val_begin_idx = opt['val_begin_idx']
        test_begin_idx = opt['test_begin_idx']
    split = np.zeros(len(obj_data), dtype=np.int32)
    assert len(obj_data) == len(data_split)
    for i, info in enumerate(rel_data):
        splitix = 0
        if opt is None: # use encode from input file
            s = data_split[i]
            splitix = s
            # if s == 'val': splitix = 1
            # if s == 'test': splitix = 2
        else: # use portion split
            if i >= val_begin_idx: splitix = 1
            if i >= test_begin_idx: splitix = 2
        split[i] = splitix
    if opt is not None and opt['shuffle']:
        np.random.shuffle(split)
    print(('assigned %d/%d/%d to train/val/test split' % (np.sum(split==0), np.sum(split==1), np.sum(split==2))))
    return split
def encode_splits(obj_data, rel_data, base_predicate, data_split, opt=None):

    split = np.zeros(len(rel_data), dtype=np.int32)
    if opt is not None:
        val_begin_idx = opt['val_begin_idx']
        test_begin_idx = opt['test_begin_idx']
    split = np.zeros(len(obj_data), dtype=np.int32)
    assert len(obj_data) == len(data_split)
    for i, info in enumerate(rel_data):
        splitix = 0
        if opt is None: # use encode from input file
            s = data_split[i]
            splitix = s
            # if s == 'val': splitix = 1
            # if s == 'test': splitix = 2
        else: # use portion split
            if i >= val_begin_idx: splitix = 1
            if i >= test_begin_idx: splitix = 2
        split[i] = splitix
    # filter test set and valid set(keep both coarse-grained and fine-grained)
    # for open-vocabulary, we need evaluate both base classes and novel classes, thus we dont need to filter
    remain_num = 0
    total_num = 0 
    for i, rel_info in enumerate(rel_data):
        if split[i] != 2:
            id_to_idx = obj_data[i]['id_to_idx']
            rel_remain = []
            for r in rel_info['relationships']:
                # only preserve base predicates for train and valid
                if r['predicate'] in base_predicate:
                    rel_remain.append(r)
                    remain_num += 1
                total_num += 1
            rel_info['relationships'] = rel_remain
            rel_data[i] = rel_info
    print('%i rel remain in train-set and valid-set of %i' % (remain_num, total_num))
    
    if opt is not None and opt['shuffle']:
        np.random.shuffle(split)

    print(('assigned %d/%d/%d to train/val/test split' % (np.sum(split==0), np.sum(split==1), np.sum(split==2))))
    return split


def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab


def make_list(list_file):
    """create a blacklist list from a file"""
    return [line.strip('\n').strip('\r') for line in open(list_file)]


def filter_object_boxes(data, heights, widths, area_frac_thresh):
    """
    filter boxes by a box area-image area ratio threshold
    """
    thresh_count = 0
    all_count = 0
    for i, img in enumerate(data):
        filtered_obj = []
        area = float(heights[i]*widths[i])
        for obj in img['objects']:
            if float(obj['h'] * obj['w']) > area * area_frac_thresh:
                filtered_obj.append(obj)
                thresh_count += 1
            all_count += 1
        img['objects'] = filtered_obj
    print('box threshod: keeping %i/%i boxes' % (thresh_count, all_count))

def filter_by_idx(data, valid_list):
    return [data[i] for i in valid_list]


def obj_rel_cross_check(obj_data, rel_data, verbose=False):
    """
    make sure all objects that are in relationship dataset
    are in object dataset
    """
    num_img = len(obj_data)
    num_correct = 0
    total_rel = 0
    for i in range(num_img):
        assert(obj_data[i]['image_id'] == rel_data[i]['image_id'])
        objs = obj_data[i]['objects']
        rels = rel_data[i]['relationships']
        ids = [obj['object_id'] for obj in objs]
        for rel in rels:
            if rel['subject']['object_id'] in ids \
                and rel['object']['object_id'] in ids:
                num_correct += 1
            elif verbose:
                if rel['subject']['object_id'] not in ids:
                    print(str(rel['subject']['object_id']) + 'cannot be found in ' + str(i))
                if rel['object']['object_id'] not in ids:
                    print(str(rel['object']['object_id']) + 'cannot be found in ' + str(i))
            total_rel += 1
    print('cross check: %i/%i relationship are correct' % (num_correct, total_rel))


def sync_objects(obj_data, rel_data):
    num_img = len(obj_data)
    for i in range(num_img):
        assert(obj_data[i]['image_id'] == rel_data[i]['image_id'])
        objs = obj_data[i]['objects']
        rels = rel_data[i]['relationships']

        ids = [obj['object_id'] for obj in objs]
        for rel in rels:
            if rel['subject']['object_id'] not in ids:
                rel_obj = rel['subject']
                rel_obj['names'] = [rel_obj['name']]
                objs.append(rel_obj)
            if rel['object']['object_id'] not in ids:
                rel_obj = rel['object']
                rel_obj['names'] = [rel_obj['name']]
                objs.append(rel_obj)

        obj_data[i]['objects'] = objs


def main(args):
    save_path = f'class_{args.mode}'
    time_0 = time.time()
    obj_data = np.load(os.path.join(save_path,'obj_data.npy'),allow_pickle=True).tolist()
    rel_data = np.load(os.path.join(save_path,'rel_data.npy'),allow_pickle=True).tolist()
    img_data = np.load(os.path.join(save_path,'img_data.npy'),allow_pickle=True).tolist()
    base_filter_predicate = np.load(os.path.join(save_path,'base_filter_predicate.npy'),allow_pickle=True).item()
    base_predicate_to_idx = np.load(os.path.join(save_path,'base_predicate_to_idx.npy'),allow_pickle=True).item()
    encoded_boxes = np.load(os.path.join(save_path,'encoded_boxes.npy'),allow_pickle=True).item()
    encoded_label = np.load(os.path.join(save_path,'encoded_label.npy'),allow_pickle=True)
    expand_predicate_to_idx = np.load(os.path.join(save_path,'expand_predicate_to_idx.npy'),allow_pickle=True).item()
    expand_predicate_token_counter = np.load(os.path.join(save_path,'expand_predicate_token_counter.npy'),allow_pickle=True).item()
    idx_to_base_predicate = np.load(os.path.join(save_path,'idx_to_base_predicate.npy'),allow_pickle=True).item()
    idx_to_expand_predicate = np.load(os.path.join(save_path,'idx_to_expand_predicate.npy'),allow_pickle=True).item()
    idx_to_label = np.load(os.path.join(save_path,'idx_to_label.npy'),allow_pickle=True).item()
    idx_to_novel_predicate = np.load(os.path.join(save_path,'idx_to_novel_predicate.npy'),allow_pickle=True).item()
    idx_to_predicate = np.load(os.path.join(save_path,'idx_to_predicate.npy'),allow_pickle=True).item()
    im_to_first_obj = np.load(os.path.join(save_path,'im_to_first_obj.npy'),allow_pickle=True)
    im_to_last_obj = np.load(os.path.join(save_path,'im_to_last_obj.npy'),allow_pickle=True)
    label_to_idx = np.load(os.path.join(save_path,'label_to_idx.npy'),allow_pickle=True).item()
    novel_predicate_to_idx = np.load(os.path.join(save_path,'novel_predicate_to_idx.npy'),allow_pickle=True).item()
    object_token_counter = np.load(os.path.join(save_path,'object_token_counter.npy'),allow_pickle=True).item()
    predicate_to_idx = np.load(os.path.join(save_path,'predicate_to_idx.npy'),allow_pickle=True).item()
    predicate_token_counter = np.load(os.path.join(save_path,'predicate_token_counter.npy'),allow_pickle=True).item()
    split = np.load(os.path.join(save_path,'split.npy'),allow_pickle=True)
    time_1 = time.time()
    print('加载数据耗时:',time_1-time_0)
    # return
    expand_number, new_predicate_dict = expand_relationships(rel_data, obj_data, img_data, split, encoded_label, idx_to_label, im_to_first_obj, im_to_last_obj, encoded_boxes, expand_predicate_to_idx)
    print('expand relationships: ', expand_number)
    print('new_predicate_dict: ', new_predicate_dict)
    # update expand_predicate_token_counter by new predicates
    for new_key in new_predicate_dict.keys():
        if new_key in expand_predicate_token_counter:
            expand_predicate_token_counter[new_key] += new_predicate_dict[new_key]
        else:
            expand_predicate_token_counter[new_key] = new_predicate_dict[new_key]
    # for open-vocabulary update rel_token_id with rel_dict
    encoded_predicate, encoded_rel, im_to_first_rel, im_to_last_rel, filter_predicate_counter = \
    encode_relationships(rel_data, expand_predicate_to_idx, obj_data, new_predicate_dict, expand_predicate_token_counter)
    # update number of relationships
    f = h5.File(os.path.join(save_path, args.h5_file), 'a')
    f.create_dataset('predicates', data=encoded_predicate)
    f.create_dataset('relationships', data=encoded_rel)
    f.create_dataset('img_to_first_rel', data=im_to_first_rel)
    f.create_dataset('img_to_last_rel', data=im_to_last_rel)

    print('new num objects = %i' % encoded_label.shape[0])
    print('new num relationships = %i' % encoded_predicate.shape[0])
    # and write the additional json file
    json_struct = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'target_to_idx': novel_predicate_to_idx,
        'idx_to_target': idx_to_novel_predicate,
        'base_to_idx': base_predicate_to_idx,
        'idx_to_base': idx_to_base_predicate,
        'predicate_to_idx': expand_predicate_to_idx,
        'idx_to_predicate': idx_to_expand_predicate,
        'predicate_count': expand_predicate_token_counter,
        'filter_predicate_counter': filter_predicate_counter,
        'object_count': object_token_counter,
        'new_predicate_count': new_predicate_dict
    }
    with open(os.path.join(save_path, args.json_file), 'w') as f:
        json.dump(json_struct, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imdb', default='./imdb_512.h5', type=str)
    parser.add_argument('--object_input', default='datasets/vg/objects.json', type=str)
    parser.add_argument('--relationship_input', default='datasets/vg/relationships.json', type=str)
    parser.add_argument('--metadata_input', default="datasets/vg/image_data.json", type=str)
    parser.add_argument('--object_alias', default='datasets/vg/object_alias.txt', type=str)
    parser.add_argument('--pred_alias', default='datasets/vg/predicate_alias.txt', type=str)
    parser.add_argument('--object_list', default='datasets/vg/object_list.txt', type=str)
    parser.add_argument('--pred_list', default='datasets/vg/predicate_list.txt', type=str)
    parser.add_argument('--input_split_file', default="datasets/vg/VG-SGG.h5")
    parser.add_argument('--num_objects', default=150, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--num_predicates', default=50, type=int, help="set to 0 to disable filtering")
    parser.add_argument('--min_box_area_frac', default=0.002, type=float)
    
    parser.add_argument('--load_frac', default=1, type=float)
    parser.add_argument('--use_input_split', default=True, type=bool)
    parser.add_argument('--train_frac', default=0.7, type=float)
    parser.add_argument('--val_frac', default=0.8, type=float)
    parser.add_argument('--shuffle', default=False, type=bool)
    parser.add_argument('--mode', default='50', type=str,choices=['50','all'])
    parser.add_argument('--h5_file', default='VG-SGG-base-EXPANDED.h5')         #写入
    parser.add_argument('--h5_file_base', default='VG-SGG-base.h5')
    parser.add_argument('--json_file_base', default='VG-SGG-base-dicts.json')
    parser.add_argument('--json_file', default='VG-SGG-base-EXPANDED-dicts.json')
    args = parser.parse_args()
    target_words = [line.strip('\n').strip('\r') for line in open('datasets/vg/predicate_list.txt')]
    mapping_dict = json.load(open('utils_data/mapping/openworld_predicate_mapping_dict_50.json'))
    if args.mode == '50':
        _, _, words = load_vg_mapping_dataset_image_text('datasets/image_caption_triplet.json')
    else:
        _, _, words = load_vg_dataset_image_text('datasets/image_caption_triplet_all.json')
    device = 'cuda'
    main(args)
