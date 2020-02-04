from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import json
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import Dataset
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap
from dae import *
from editnet import *

class COCOTestDataset(Dataset):

    def __init__(self):

        self.val_hf = h5py.File('bottom-up features' + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']
        self.cpi = 5
        
        with open('caption data/TEST_names_coco.json', 'r') as j:
            self.names = json.load(j)
            
        with open('caption data/CAPUTIL_test.json', 'r') as j:
            self.caption_util = json.load(j)
            
        with open(os.path.join('caption data',  'TEST_GENOME_DETS_coco.json'), 'r') as j:
            self.objdet = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.names)

    def __getitem__(self, i):

        img_name = self.names[i]
        objdet = self.objdet[i]

        previous_caption = torch.LongTensor(self.caption_util[img_name]['encoded_previous_caption'])
        image_id = torch.LongTensor([self.caption_util[img_name]['image_ids']])
        prev_caplen = torch.LongTensor(self.caption_util[img_name]['previous_caption_length'])
        assert objdet[0] == "v"
        img = torch.FloatTensor(self.val_features[objdet[1]])
        
        return img, image_id, previous_caption, prev_caplen

    def __len__(self):
        return self.dataset_size


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True 
dae_checkpoint = 'BEST_checkpoint_12.pth.tar'
editnet_checkpoint = 'BEST_checkpoint_38.pth.tar'
annFile = 'cococaption/annotations/captions_val2014.json'
emb_file = 'glove.6B.300d.txt'

with open('caption data/WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)
    
rev_word_map = {v: k for k, v in word_map.items()}

test_loader = torch.utils.data.DataLoader(COCOTestDataset(),
                                          batch_size = 1,
                                          shuffle=True, 
                                          pin_memory=True)

dae_ckpt = torch.load(dae_checkpoint)
dae_ar = dae_ckpt['dae_ar']
dae_ar = dae_ar.to(device)

for param in dae_ar.affine_hidden.parameters():
    param.requires_grad = False
    
editnet_ckpt = torch.load(editnet_checkpoint)
decoder = editnet_ckpt['decoder']
decoder = decoder.to(device)


def evaluate_full(loader, dae_ar, decoder, beam_size, epoch, word_map):
    
    vocab_size = len(word_map)
    decoder.eval()
    dae_ar.eval()
    results = []
    rev_word_map = {v: k for k, v in word_map.items()}
    
    for i, (img, image_id, previous_caption, prev_caplen) in enumerate(tqdm(loader, 
                                                                        desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        k = beam_size
        infinite_pred = False
        
        image_features = img.to(device)  
        image_id = image_id.to(device)  # (1,1)
        encoded_previous_captions = previous_caption.to(device) 
        previous_cap_length = prev_caplen.to(device) 
        img_mean = image_features.mean(1)
        
        eprevious_encoded_h, eprevious_encoded_m, efinal_hidden, eprev_cap_mask = decoder.caption_encoder(encoded_previous_captions, 
                                                                                                          previous_cap_length)
        dprevious_encoded, dfinal_hidden, dprev_caption_mask = dae_ar.dae.caption_encoder(encoded_previous_captions, 
                                                                                          previous_cap_length)
        image_features = image_features.expand(k, -1, -1)
        img_mean = img_mean.expand(k, -1)
        eprevious_encoded_h = eprevious_encoded_h.expand(k, -1, -1)
        eprevious_encoded_m = eprevious_encoded_m.expand(k, -1, -1)
        efinal_hidden = efinal_hidden.expand(k, -1)
        eprev_cap_mask = eprev_cap_mask.expand(k, -1)
        dprevious_encoded = dprevious_encoded.expand(k, -1, -1)
        dprev_cap_mask = dprev_caption_mask.expand(k, -1)
        dfinal_hidden = dfinal_hidden.expand(k,-1)
        
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        seqs = k_prev_words  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        complete_seqs = list()
        complete_seqs_scores = list()
        step = 1
        eh1, ec1 = decoder.init_hidden_state(k)  # (k, decoder_dim)
        eh2, ec2 = decoder.init_hidden_state(k)  # (k, decoder_dim)
        dh1, dc1 = dae_ar.dae.init_hidden_state(k)  # (batch_size, decoder_dim)
        dh2, dc2 = dae_ar.dae.init_hidden_state(k)
        
        while True:
            eembeddings = decoder.embed(k_prev_words).squeeze(1)
            etopdown_input = torch.cat([eembeddings, efinal_hidden, eh2, img_mean], dim=1)
            eh1, ec1 = decoder.attention_lstm(etopdown_input, (eh1, ec1))
            eattend_cap, ealpha_c = decoder.caption_attention(eprevious_encoded_h, eh1, eembeddings, eprev_cap_mask)
            eattend_img = decoder.visual_attention(image_features, eh1)
            elanguage_input = torch.cat([eh1, eattend_cap, eattend_img], dim = 1)
            eselected_memory = decoder.select(eprevious_encoded_m, ealpha_c)
            eh2,ec2 = decoder.copy_lstm(elanguage_input, (eh2, ec2), eselected_memory)
            escores = decoder.fc(eh2)
            ########################################################################################
            dembeddings = dae_ar.dae.embed(k_prev_words).squeeze(1)        
            dtopdown_input = torch.cat([dembeddings, dfinal_hidden, dh2],dim=1)
            dh1,dc1 = dae_ar.dae.attention_lstm(dtopdown_input, (dh1, dc1))
            dattend_cap = dae_ar.dae.caption_attention(dprevious_encoded, dh1, dprev_cap_mask)
            dlanguage_input = torch.cat([dh1, dattend_cap], dim = 1)
            dh2,dc2 = dae_ar.dae.language_lstm(dlanguage_input, (dh2, dc2))
            dscores = dae_ar.dae.fc(dh2)  
            ########################################################################################
            soft_escores = F.softmax(escores, dim = 1)
            soft_dscores = F.softmax(dscores, dim = 1)
            scores = ((soft_escores + soft_dscores)/2).log()
            ########################################################################################
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
                
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  
            
            if k == 0:
                break
                
            seqs = seqs[incomplete_inds]
            eh1 = eh1[prev_word_inds[incomplete_inds]]
            ec1 = ec1[prev_word_inds[incomplete_inds]]
            eh2 = eh2[prev_word_inds[incomplete_inds]]
            ec2 = ec2[prev_word_inds[incomplete_inds]]
            image_features = image_features[prev_word_inds[incomplete_inds]]
            img_mean = img_mean[prev_word_inds[incomplete_inds]]
            efinal_hidden = efinal_hidden[prev_word_inds[incomplete_inds]]
            eprevious_encoded_h = eprevious_encoded_h[prev_word_inds[incomplete_inds]]
            eprevious_encoded_m = eprevious_encoded_m[prev_word_inds[incomplete_inds]]
            eprev_cap_mask = eprev_cap_mask[prev_word_inds[incomplete_inds]]
            ###########################################################################################
            dh1 = dh1[prev_word_inds[incomplete_inds]]
            dc1 = dc1[prev_word_inds[incomplete_inds]]
            dh2 = dh2[prev_word_inds[incomplete_inds]]
            dc2 = dc2[prev_word_inds[incomplete_inds]]
            dprevious_encoded = dprevious_encoded[prev_word_inds[incomplete_inds]]
            dprev_cap_mask = dprev_cap_mask[prev_word_inds[incomplete_inds]]
            dfinal_hidden = dfinal_hidden[prev_word_inds[incomplete_inds]]
            ###########################################################################################
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            
            if step > 50:
                infinite_pred = True
                break
            step += 1
            
        if infinite_pred is not True:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0][:18]
            seq = [seq[i].item() for i in range(len(seq))]
            
        # Construct Sentence
        sen_idx = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        sentence = ' '.join([rev_word_map[sen_idx[i]] for i in range(len(sen_idx))])
        item_dict = {"image_id": image_id.item(), "caption": sentence}
        results.append(item_dict)
        
    print("Calculating Evalaution Metric Scores......\n")
    resFile = 'cococaption/results/captions_val2014_results_' + str(epoch) + '.json' 
    evalFile = 'cococaption/results/captions_val2014_eval_' + str(epoch) + '.json' 
    # Calculate Evaluation Scores
    with open(resFile, 'w') as wr:
        json.dump(results,wr)
        
    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)
    # evaluate on a subset of images
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    # evaluate results
    cocoEval.evaluate()    
    # Save Scores for all images in resFile
    with open(evalFile, 'w') as w:
        json.dump(cocoEval.eval, w)

    return cocoEval.eval['CIDEr'], cocoEval.eval['Bleu_4']


evaluate_full(loader = test_loader, 
              dae_ar = dae_ar, 
              decoder = decoder, 
              beam_size = 3, 
              epoch = 0, 
              word_map = word_map)