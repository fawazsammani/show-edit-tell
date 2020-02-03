import json
import time
import numpy as np
import torch

splits = ['train', 'val', 'test']
beams = [1,3,3]
max_len = 18

for split,beam in zip(splits,beams):
    with open('aoa_caps/results_' + split + '_beam ` + str(beam).json', 'r') as j:
        captions = json.load(j)

    with open('caption data/WORDMAP_coco.json', 'r') as r:
        word_map = json.load(r)

    caps_dic = {}
    for item in captions:
        img_name = item['file_name'].split('\\')[1]
        caps_dic[img_name] = {}
        cap = item['caption']
        c = cap.split()
        enc_c = [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<pad>']] * (max_len - len(c))
        mask = [1 if w!=0 else 0 for w in enc_c]
        caps_dic[img_name]['caption'] = cap
        caps_dic[img_name]['encoded_previous_caption'] = enc_c
        caps_dic[img_name]['previous_caption_length'] = [len(c)]
        caps_dic[img_name]['image_ids'] = item['image_id']


    with open('caption data/CAPUTIL_' + split + '.json', 'w') as w:
        json.dump(caps_dic, w)



