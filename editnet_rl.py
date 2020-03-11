import os
import numpy as np
import json
import torch
import h5py
import math
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from cococaption.pycocotools.coco import COCO
from cococaption.pycocoevalcap.eval import COCOEvalCap

class COCOTrainDataset(Dataset):

    def __init__(self):
        
        # Open hdf5 file where images are stored
        self.train_hf = h5py.File('bottom-up features' + '/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File('bottom-up features' + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']
        self.cpi = 5

        with open(os.path.join('caption data','TRAIN_CAPTIONS_coco.json'), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join('caption data', 'TRAIN_CAPLENS_coco.json'), 'r') as j:
            self.caplens = json.load(j)
        
        with open('caption data/TRAIN_names_coco.json', 'r') as j:
            self.names = json.load(j)
            
        with open('caption data/CAPUTIL_train.json', 'r') as j:
            self.caption_util = json.load(j)
            
        with open(os.path.join('caption data', 'TRAIN_GENOME_DETS_coco.json'), 'r') as j:
            self.objdet = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        """
        returns:
        img: the image features of shape (batch_size,36, 2048)
        caption: the ground-truth caption of shape (batch_size, max_length)
        caplen: the valid length (without padding) of the ground-truth caption of shape (batch_size,1)
        previous_caption: the encoded caption of the previous model of shape (batch_size, max_length)
        previous_caption_length: the valid length (without padding) of the previous caption of shape (batch_size,1)
        """
        # The Nth caption corresponds to the (N // captions_per_image)th image
        img_name = self.names[i // self.cpi]
        objdet = self.objdet[i // self.cpi]

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        
        if objdet[0] == "v":
            img = torch.FloatTensor(self.val_features[objdet[1]])
        else:
            img = torch.FloatTensor(self.train_features[objdet[1]])
        
        previous_caption = torch.LongTensor(self.caption_util[img_name]['encoded_previous_caption'])
        prev_caplen = torch.LongTensor(self.caption_util[img_name]['previous_caption_length'])
        # We also need the 5 captions for an image when training with self-critical (used in cider score calculation) 
        all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        
        return img, caption, caplen, previous_caption, prev_caplen, all_captions

    def __len__(self):
        return self.dataset_size
    
    
class COCOValidationDataset(Dataset):

    def __init__(self):

        self.val_hf = h5py.File('bottom-up features' + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']
        self.cpi = 5
        
        with open('caption data/VAL_names_coco.json', 'r') as j:
            self.names = json.load(j)
            
        with open('caption data/CAPUTIL_val.json', 'r') as j:
            self.caption_util = json.load(j)
            
        with open(os.path.join('caption data',  'VAL_GENOME_DETS_coco.json'), 'r') as j:
            self.objdet = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.names)

    def __getitem__(self, i):
        """
        returns:
        img: the image features of shape (batch_size,36, 2048)
        previous_caption: the encoded caption of the previous model of shape (batch_size, max_length)
        image_id: the respective id for the image of shape (batch_size, 1)
        previous_caption_length: the valid length (without padding) of the previous caption of shape (batch_size,1)
        """
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
        """
        returns:
        img: the image features of shape (batch_size,36, 2048)
        previous_caption: the encoded caption of the previous model of shape (batch_size, max_length)
        image_id: the respective id for the image of shape (batch_size, 1)
        previous_caption_length: the valid length (without padding) of the previous caption of shape (batch_size,1)
        """
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


def save_checkpoint(epoch, epochs_since_improvement, decoder, decoder_optimizer, cider, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'cider': cider,
             'decoder': decoder,
             'decoder_optimizer': decoder_optimizer}
    
    filename = 'checkpoint_' + str(epoch) + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum() 
    return correct_total.item() * (100.0 / batch_size)

def adjust_learning_rate(optimizer, shrink_factor):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
    
def set_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class LSTMCellC(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCellC, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.tanh = nn.Tanh()
        self.init_parameters()
    
    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-std, std)

    def forward(self, x, states):
        """
        inp shape: (batch_size, input_size)
        each of states shape: (batch_size, hidden_size)
        """
        
        ht, ct = states
        gates = self.x2h(x) + self.h2h(ht)    # (batch_size, 4 * hidden_size)

        in_gate, forget_gate, new_memory, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        new_memory = self.tanh(new_memory)
        c_new = (forget_gate * ct) + (in_gate * new_memory)
        h_new = out_gate * self.tanh(c_new)

        return h_new, c_new


class CopyLSTMCellC(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(CopyLSTMCellC, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.gate_cnew = nn.Linear(hidden_size, hidden_size)
        self.gate_cmem = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.init_parameters()
    
    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-std, std)

    def forward(self, x, states, c_memory):
        """
        inp shape: (batch_size, input_size)
        each of states shape: (batch_size, hidden_size)
        encoder_memory shape: (batch_size, hidden_size)
        """
        ht, ct = states
        gates = self.x2h(x) + self.h2h(ht)    # (batch_size, 5 * hidden_size)

        in_gate, forget_gate, new_memory, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        new_memory = self.tanh(new_memory)
        c_new = (forget_gate * ct) + (in_gate * new_memory)
        copy_gate = torch.sigmoid(self.gate_cnew(c_new) + self.gate_cmem(c_memory))
        adaptive_memory = (copy_gate * c_memory) + (1 - copy_gate) * c_new
        h_new = out_gate * self.tanh(adaptive_memory)

        return h_new, adaptive_memory


class EmbeddingC(nn.Module):

    def __init__(self, word_map, emb_dim):

        super(EmbeddingC, self).__init__()
        
        self.emb_dim = emb_dim
        self.word_map = word_map
        self.embedding = nn.Embedding(len(word_map), self.emb_dim)  # embedding layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        
    def forward(self, x):
        out = self.embedding(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class CaptionEncoderC(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, embed):
        super(CaptionEncoderC, self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.embed = embed
        self.lstm_encoder_cell = LSTMCellC(emb_dim, enc_hid_dim)
        self.affine_hn = nn.Linear(enc_hid_dim, enc_hid_dim)
        self.tanh = nn.Tanh()
        
    def forward(self, seq, seq_len):
        
        batch_size = seq.size(0)
        sorted_lengths, sort_indices = seq_len.squeeze(1).sort(dim=0, descending=True)
        inv_ix = sort_indices.clone()
        inv_ix[sort_indices] = torch.arange(0,len(sort_indices)).type_as(inv_ix)
        sorted_lengths = sorted_lengths.tolist()
        sorted_sequences = seq[sort_indices]
        hidden_states = torch.zeros(batch_size, max(sorted_lengths), self.enc_hid_dim).to(device)
        memory_states = torch.zeros(batch_size, max(sorted_lengths), self.enc_hid_dim).to(device)
        final_hidden = torch.zeros(batch_size, self.enc_hid_dim).to(device)
        h,c = [torch.zeros(batch_size, self.enc_hid_dim).to(device), torch.zeros(batch_size, self.enc_hid_dim).to(device)]
        embeddings = self.embed(sorted_sequences)

        for t in range(max(sorted_lengths)):
            batch_size_t = sum([l > t for l in sorted_lengths])
            h, c = self.lstm_encoder_cell(embeddings[:batch_size_t, t, :], (h[:batch_size_t], c[:batch_size_t])) 
            hidden_states[:batch_size_t, t, :] = h.clone()
            memory_states[:batch_size_t, t, :] = c.clone()
            final_hidden[:batch_size_t] = h.clone()
            
        mask = ((memory_states.sum(2))!=0).float()
        final_hidden = self.tanh(self.affine_hn(final_hidden))
        
        hidden_states = hidden_states[inv_ix]
        memory_states = memory_states[inv_ix]
        final_hidden = final_hidden[inv_ix]
        mask = mask[inv_ix]
        
        return hidden_states, memory_states, final_hidden, mask


class CaptionAttentionC(nn.Module):

    def __init__(self, caption_features_dim, decoder_dim, attention_dim):

        super(CaptionAttentionC, self).__init__()
        self.cap_features_att = nn.Linear(caption_features_dim, attention_dim) 
        self.cap_decoder_att = nn.Linear(decoder_dim, attention_dim) 
        self.cap_full_att = nn.Linear(attention_dim, 1)
        self.context_gate = nn.Linear((caption_features_dim * 2) + decoder_dim, caption_features_dim)
        self.sc_affine = nn.Linear(caption_features_dim, caption_features_dim)
        self.tc_affine = nn.Linear(decoder_dim * 2, caption_features_dim)
        self.tanh = nn.Tanh()

    def forward(self, caption_features, decoder_hidden, word, prev_caption_mask):
        """
        caption features of shape: (batch_size, max_seq_length, caption_features_dim)
        prev_caption_mask of shape: (batch_size, max_seq_length)
        decoder_hidden is the current output of the decoder LSTM of shape (batch_size, decoder_dim)
        """
        att1_c = self.cap_features_att(caption_features)  # (batch_size, max_words, attention_dim)
        att2_c = self.cap_decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att_c = self.cap_full_att(self.tanh(att1_c + att2_c.unsqueeze(1))).squeeze(2)  # (batch_size, max_words)
        # Masking for zero pads for attention computation
        att_c = att_c.masked_fill(prev_caption_mask == 0, -1e10)   # (batch_size, max_words) * (batch_size, max_words)
        alpha_c = F.softmax(att_c, dim = 1)  # (batch_size, max_words)
        context = (caption_features * alpha_c.unsqueeze(2)).sum(dim=1)  # (batch_size, caption_features_dim)
        # Context Gating
        zt = torch.sigmoid(self.context_gate(torch.cat([word, decoder_hidden, context], dim = 1)))
        tc_input = torch.cat([word, decoder_hidden], dim = 1)
        gated_context = zt * self.tanh(self.sc_affine(context)) + (1 - zt) * self.tanh(self.tc_affine(tc_input))
        return gated_context , alpha_c

class SelectC(nn.Module):
    """
    SCMA Mechanism
    The code below includes the mechanism as discussed in the paper. However, implementation-wise, there is a simpler way, which is 
    filling the unwanted scores with -inf before the softmax operation. By running softmax on all -inf scores except the maximum one,
    you can get the same output. To implement this, pass the scores (before softmax) rather than passing the softamx weights to the 
    forward function, and perform the following:
    
    scores_c = scores.detach()
    value, max_indices = torch.max(scores_c,1)        # (batch_size)
    value = value.unsqueeze(1)                # (batch_size,1)
    mask = torch.zeros_like(scores_c)       # (batch_size, words)
    mask.scatter_(1, max_indices.unsqueeze(1), 1)
    scores = scores.masked_fill(mask == 0, -float("inf"))
    sim_weights = F.softmax(logits, dim = -1)
    selected_memory = (sim_weights.unsqueeze(2) * previous_encoded_m).sum(dim = 1)
    """
    def __init__(self, prev_caption_dim, decoder_dim):
        super(SelectC, self).__init__()
        
    def forward(self, previous_encoded_m, sim_weights, soft = False):
        """
        previous_encoded_c of shape (batch_size, max_words, 1024)
        sim_weights os shape (batch_size, max_words)
        soft: use soft attention or non-differentiable indexing?
        """
        if not soft:
            sim_weights_c = sim_weights.detach()
            value, max_indices = torch.max(sim_weights_c,1)        # (batch_size)
            value = value.unsqueeze(1)                # (batch_size,1)
            mask = torch.zeros_like(sim_weights_c)       # (batch_size, words)
            mask.scatter_(1, max_indices.unsqueeze(1), 1)
            mask_diff = mask.clone()
            values_in_batch = value.squeeze(1)
            mask_diff[mask_diff == 1] = 1 - values_in_batch 
            sim_weights = (sim_weights * mask) + mask_diff      # (batch_size, max_words)
            
        selected_memory = (sim_weights.unsqueeze(2) * previous_encoded_m).sum(dim = 1)
        return selected_memory

    
class VisualAttentionC(nn.Module):

    def __init__(self, image_features_dim, decoder_dim, attention_dim):

        super(VisualAttentionC, self).__init__()
        
        self.att_embed = nn.Sequential(nn.Linear(image_features_dim, decoder_dim),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
        
        self.features_att = nn.Linear(decoder_dim, attention_dim) 
        self.decoder_att = nn.Linear(decoder_dim, attention_dim) 
        self.full_att = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, image_features, decoder_hidden):
        
        features_embed = self.att_embed(image_features)   # (batch_size, 36, 1024)
        att1 = self.features_att(features_embed)  # (batch_size, 36, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(F.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, 36)
        alpha = self.softmax(att)  # (batch_size, 36)
        context = (image_features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, 2048)
        return context


class DecoderC(nn.Module):

    def __init__(self, 
                 word_map, 
                 decoder_dim = 1024, 
                 caption_features_dim = 1024, 
                 emb_dim = 1024, 
                 attention_dim = 512, 
                 image_features_dim = 2048):

        super(DecoderC, self).__init__()
        self.vocab_size = len(word_map)
        self.dropout = nn.Dropout(0.5)
        self.decoder_dim = decoder_dim
        self.embed = EmbeddingC(word_map, emb_dim)
        self.caption_encoder = CaptionEncoderC(len(word_map), emb_dim, caption_features_dim, self.embed)
        self.caption_attention = CaptionAttentionC(caption_features_dim, decoder_dim, attention_dim)
        self.visual_attention = VisualAttentionC(image_features_dim, decoder_dim, attention_dim)
        self.select = SelectC(caption_features_dim, decoder_dim)
        self.attention_lstm = nn.LSTMCell((emb_dim * 3) + image_features_dim, decoder_dim)
        self.copy_lstm = CopyLSTMCellC((emb_dim * 2) + image_features_dim, decoder_dim)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(decoder_dim, self.vocab_size)
        
    def init_hidden_state(self,batch_size):

        h = torch.zeros(batch_size,self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
        return h, c

    def forward(self, word_map, encoded_previous_captions, previous_cap_length, image_features, sample_max, sample_rl):
        
        max_len = 18
        batch_size = image_features.size(0)

        seq = torch.zeros(batch_size, max_len, dtype=torch.long).to(device)
        seqLogprobs = torch.zeros(batch_size, max_len).to(device)

        start_idx = word_map['<start>']
        it = torch.LongTensor(batch_size).to(device)   # (batch_size) 
        it[:] = start_idx

        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        previous_encoded_h, previous_encoded_m, final_hidden, prev_cap_mask = self.caption_encoder(encoded_previous_captions, 
                                                                                                   previous_cap_length)
        image_mean = image_features.mean(1)
        
        for timestep in range(max_len + 1):
            
            embeddings = self.embed(it)       # (batch_size, embed_dim)
            topdown_input = torch.cat([embeddings, final_hidden, h2, image_mean],dim=1)
            h1, c1 = self.attention_lstm(topdown_input, (h1, c1))
            attend_cap, alpha_c = self.caption_attention(previous_encoded_h, h1, embeddings, prev_cap_mask) 
            attend_img = self.visual_attention(image_features, h1) 
            language_input = torch.cat([h1, attend_cap, attend_img], dim = 1)
            selected_memory = self.select(previous_encoded_m, alpha_c)
            h2,c2 = self.copy_lstm(language_input, (h2, c2), selected_memory)
            pt = self.fc(self.dropout(h2))
            logprobs = F.log_softmax(pt, dim=1)   # (batch_size, vocab_size)

            # if we reached to the maximum length, stop sampling and leave the 0 in the last element (as initialized)
            if timestep == max_len:
                break

            if sample_max: # Greedy decoding
                sampleLogprobs, it = torch.max(logprobs, 1)
                it = it.view(-1).long()

            if sample_rl:   # Sampling from multinomial for self-critical
                prob_prev = torch.exp(logprobs)     # fetch prev distribution (softmax)
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # flatten indices for saving in tensor
                
            # Replace <end> token (if there is) with 0. Otherwise, a lot to change in ruotianluo code
            it = it.clone()
            it[it == word_map['<end>']] = 0

            # If all batches predict the <end> token, then stop looping
            if timestep == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
                
            it = it * unfinished.type_as(it)
            
            seq[:,timestep] = it
            seqLogprobs[:,timestep] = sampleLogprobs.view(-1)
            
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        
        return seq, seqLogprobs



class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, sample_logprobs, seq, reward):
        
        sample_logprobs = sample_logprobs.view(-1)   # (batch_size * max_len)
        reward = reward.view(-1)
        # set mask elements for all <end> tokens to 0 
        mask = (seq>0).float()                        # (batch_size, max_len)
        
        # account for the <end> token in the mask. We do this by shifting the mask one timestep ahead
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)
        
        if not mask.is_contiguous():
            mask = mask.contiguous()
        
        mask = mask.view(-1)
        output = - sample_logprobs * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output


import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("coco-caption")

CiderD_scorer = None

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    
def preprocess_gd(allcaps, word_map):
    """
    allcaps: Long tensor of shape (batch_size, 5, max_len)
    """
    ground_truth = []
    for j in range(allcaps.shape[0]):
        # when training with RL, no need to sort the batches as we did in cross-entropy training, since we don't feed
        # the ground truth encoded captions to the LSTM language model
        img_caps = allcaps[j].tolist()   # list of length 5
        img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}], img_caps)) 
        # 0 will get removed later in array_to_str
        img_captions_z = list(map(lambda c:[w if w!=word_map['<end>'] else 0 for w in c], img_captions)) 
        ground_truth.append(img_captions_z)
    return ground_truth  # list of length batch_size, each element in this list contains the 5 captions in another list (3D list)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        # If reached end token
        if arr[i] == 0:   # not word_map['<end>']. Remember we replaced word_map['<end>'] with 0 in the sample function
            break
    return out.strip()

def get_self_critical_reward(gen_result, greedy_res, ground_truth, cider_weight = 1):
    
    # ground_truth is the 5 ground truth captions for a mini-batch, which can be aquired from the preprocess_gd function
    #[[c1, c2, c3, c4, c5], [c1, c2, c3, c4, c5],........]. Note that c is a caption placed in a list
    # len(ground_truth) = batch_size. Already duplicated the ground truth captions in dataloader
    
    batch_size = gen_result.size(0)  
    
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()   # (batch_size, max_len)
    greedy_res = greedy_res.data.cpu().numpy()   # (batch_size, max_len)
    
    for i in range(batch_size):
        # change to string for evaluation purpose 
        res[i] = [array_to_str(gen_result[i])]
        
    for i in range(batch_size):
        # change to string for evaluation purpose
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(ground_truth)):
        gts[i] = [array_to_str(ground_truth[i][j]) for j in range(len(ground_truth[i]))]
    
    # 2 is because one is for the sampling and one for greedy decoding
    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)] 
    # the number of ground-truth captions for each image stay the same as above. Duplicate for the sampling and greedy
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)

    scores = cider_weight * cider_scores
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)    # gen_result.shape[1] = max_len
    rewards = torch.from_numpy(rewards).float()

    return rewards


def train(train_loader, decoder, criterion, decoder_optimizer, epoch, word_map):

    decoder.train()  # train mode (dropout is used)
    sum_rewards = 0
    count = 0

    for i, (img, _, _, previous_caption, prev_caplen, allcaps) in enumerate(train_loader):
        
        samples = img.shape[0]
    
        image_features = img.to(device)
        previous_caption = previous_caption.to(device)
        prev_caplen = prev_caplen.to(device)

        decoder_optimizer.zero_grad()
        
        decoder.eval()
        with torch.no_grad():
            greedy_res, _ = decoder(word_map, previous_caption, prev_caplen, image_features, 
                                    sample_max = True, sample_rl = False)
        decoder.train()
        seq_gen, seqLogprobs = decoder(word_map, previous_caption, prev_caplen, image_features, 
                                       sample_max = False, sample_rl = True)
        
        ground_truth = preprocess_gd(allcaps, word_map)
        rewards = get_self_critical_reward(seq_gen, greedy_res, ground_truth, cider_weight = 1)
        loss = criterion(seqLogprobs, seq_gen, rewards.to(device))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)
        decoder_optimizer.step()
        
        sum_rewards += torch.mean(rewards[:,0]) * samples
        count += samples

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{}][{}/{}]\tAverage Reward: {:.3f}'.format(epoch, i, len(train_loader), sum_rewards/count))


def evaluate(loader, decoder, beam_size, epoch, vocab_size, word_map):
    
    decoder.eval()
    results = []
    rev_word_map = {v: k for k, v in word_map.items()}
    
    # For each image
    for i, (img, image_id, previous_caption, prev_caplen) in enumerate(tqdm(loader, 
                                                                        desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        infinite_pred = False
        
        image_features = img.to(device)  
        image_id = image_id.to(device)  # (1,1)
        encoded_previous_captions = previous_caption.to(device) 
        previous_cap_length = prev_caplen.to(device) 
        img_mean = image_features.mean(1)
        previous_encoded_h, previous_encoded_m, final_hidden, prev_cap_mask = decoder.caption_encoder(encoded_previous_captions, 
                                                                                                      previous_cap_length)
        # Expand all
        image_features = image_features.expand(k, -1, -1)
        img_mean = img_mean.expand(k, -1)
        previous_encoded_h = previous_encoded_h.expand(k, -1, -1)
        previous_encoded_m = previous_encoded_m.expand(k, -1, -1)
        final_hidden = final_hidden.expand(k, -1)
        prev_cap_mask = prev_cap_mask.expand(k, -1)
        
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        
        h1, c1 = decoder.init_hidden_state(k)  # (k, decoder_dim)
        h2, c2 = decoder.init_hidden_state(k)  # (k, decoder_dim)
        
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embed(k_prev_words).squeeze(1) 
            topdown_input = torch.cat([embeddings, final_hidden, h2, img_mean], dim=1)
            h1, c1 = decoder.attention_lstm(topdown_input, (h1, c1))
            attend_cap, alpha_c = decoder.caption_attention(previous_encoded_h, h1, embeddings, prev_cap_mask)
            attend_img = decoder.visual_attention(image_features, h1)
            language_input = torch.cat([h1, attend_cap, attend_img], dim = 1)
            selected_memory = decoder.select(previous_encoded_m, alpha_c)
            h2,c2 = decoder.copy_lstm(language_input, (h2, c2), selected_memory)
            scores = decoder.fc(h2)
            scores = F.log_softmax(scores, dim=1)

            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
                
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            image_features = image_features[prev_word_inds[incomplete_inds]]
            img_mean = img_mean[prev_word_inds[incomplete_inds]]
            final_hidden = final_hidden[prev_word_inds[incomplete_inds]]
            previous_encoded_h = previous_encoded_h[prev_word_inds[incomplete_inds]]
            previous_encoded_m = previous_encoded_m[prev_word_inds[incomplete_inds]]
            prev_cap_mask = prev_cap_mask[prev_word_inds[incomplete_inds]]
            
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
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


# Data parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
start_epoch = 0
epochs = 30 
epochs_since_improvement = 0 
batch_size = 80
best_cider = 0.
print_freq = 100  
checkpoint = 'editnet.tar'   # load xe checkpoint
annFile = 'cococaption/annotations/captions_val2014.json' 
cached_tokens =  'coco-train-idxs'

train_loader = torch.utils.data.DataLoader(COCOTrainDataset(),
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

val_loader = torch.utils.data.DataLoader(COCOValidationDataset(),
                                         batch_size = 1,
                                         shuffle=True, 
                                         pin_memory=True)

# Read word map
with open('caption data/WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)
    
rev_word_map = {v: k for k, v in word_map.items()}

checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
epochs_since_improvement = checkpoint['epochs_since_improvement']
best_cider = checkpoint['cider']
decoder = checkpoint['decoder']
decoder_optimizer = checkpoint['decoder_optimizer']

decoder = decoder.to(device)
criterion = RewardCriterion().to(device)


for epoch in range(start_epoch, epochs):
    
    if epoch == start_epoch:   # only at the starting epoch of self-critical. Then comment out
        set_learning_rate(decoder_optimizer, 5e-5)

    if epochs_since_improvement > 0:
        adjust_learning_rate(decoder_optimizer, 0.5)
        
        
    init_scorer(cached_tokens)
        
    # One epoch's training
    train(train_loader=train_loader,
          decoder=decoder,
          criterion = criterion,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch, 
          word_map = word_map)

    # One epoch's validation
    recent_cider, recent_bleu4 = evaluate(loader = val_loader, 
                                          decoder = decoder,
                                          beam_size = 3, 
                                          epoch = epoch, 
                                          vocab_size = len(word_map), 
                                          word_map = word_map)

    # Check if there was an improvement
    is_best = recent_cider > best_cider
    best_cider = max(recent_cider, best_cider)
    if not is_best:
        epochs_since_improvement += 1
        print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
    else:
        epochs_since_improvement = 0

    # Save checkpoint
    save_checkpoint(epoch, epochs_since_improvement, decoder, decoder_optimizer, recent_cider, is_best)

