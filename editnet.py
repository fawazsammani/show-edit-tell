import os
import numpy as np
import json
import torch
import h5py
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
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
        previous_caption_embed: the ELMo Max Pooled sentence embedding of the caption from the previous model of shape (batch_size, 1024)
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
        previous_caption_embed: the ELMo Max Pooled sentence embedding of the caption from the previous model of shape (batch_size, 1024)
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

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    print("Current learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


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
    
    def __init__(self, prev_caption_dim, decoder_dim):
        super(SelectC, self).__init__()
#        self.transform_copied = nn.Linear(prev_caption_dim, decoder_dim)
#        self.tanh = nn.Tanh()
        
    def forward(self, previous_encoded_m, sim_weights, soft = False):
        """
        previous_encoded_c of shape (batch_size, max_words, 1024)
        sim_weights os shape (batch_size, max_words)
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
#        selected_memory = self.tanh(self.transform_copied(selected_memory))
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

    def forward(self, image_features, encoded_captions, caption_lengths, encoded_previous_captions, previous_cap_length, use_ss, ss_prob):
        """
        encoded captions of shape: (batch_size, max_caption_length)
        caption_lengths of shape: (batch_size, 1)
        encoded_previous_captions: encoded previous captions to be passed to the LSTM encoder of shape: (batch_size, max_caption_length)
        previous_caption_lengths of shape: (batch_size, 1)
        """
        # Sort
        batch_size = encoded_captions.size(0)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        encoded_previous_captions = encoded_previous_captions[sort_ind]
        previous_cap_length = previous_cap_length[sort_ind]  
        # Initialize LSTM states
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        # Remove <end> from lengths since we've finished generating words when we predict <end>
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device) 
        
        previous_encoded_h, previous_encoded_m, final_hidden, prev_cap_mask = self.caption_encoder(encoded_previous_captions, 
                                                                                                   previous_cap_length)
        image_mean = image_features.mean(1)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            if use_ss and t >= 1 and ss_prob > 0.0:
                sample_prob = torch.zeros(batch_size_t).uniform_(0, 1).to(device)
                sample_mask = sample_prob < ss_prob
                if sample_mask.sum() == 0:
                    it = encoded_captions[:batch_size_t, t]
                    embeddings = self.embed(it)
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = encoded_captions[:batch_size_t, t].clone()
                    prob_prev = torch.exp(predictions[:batch_size_t, t-1].detach())
                    multinom = torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind)
                    it.index_copy_(0, sample_ind, multinom)
                    embeddings = self.embed(it)
                
            else:
                it = encoded_captions[:batch_size_t, t]
                embeddings = self.embed(it)
                

            topdown_input = torch.cat([embeddings,
                                       final_hidden[:batch_size_t], 
                                       h2[:batch_size_t], 
                                       image_mean[:batch_size_t]],dim=1)
            
            h1, c1 = self.attention_lstm(topdown_input, (h1[:batch_size_t], c1[:batch_size_t]))
            
            attend_cap, alpha_c = self.caption_attention(previous_encoded_h[:batch_size_t], h1, 
                                                         embeddings, prev_cap_mask[:batch_size_t])
            
            attend_img = self.visual_attention(image_features[:batch_size_t], h1)
            
            language_input = torch.cat([h1, attend_cap, attend_img], dim = 1)
            
            selected_memory = self.select(previous_encoded_m[:batch_size_t], alpha_c)
            
            h2,c2 = self.copy_lstm(language_input, (h2[:batch_size_t], c2[:batch_size_t]), selected_memory)

            preds = self.fc(self.dropout(h2))
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind


def train(train_loader, decoder, criterion, decoder_optimizer, epoch, word_map, use_ss, ss_prob):

    decoder.train()  # train mode (dropout is used)

    losses = AverageMeter()  # loss (per word decoded)
    top3accs = AverageMeter()  # top5 accuracy

    for i, (img, caption, caplen, previous_caption, prev_caplen, _) in enumerate(train_loader):
        
        image_features = img.to(device)
        caps = caption.to(device)
        caplens = caplen.to(device)
        previous_caption = previous_caption.to(device)
        prev_caplen = prev_caplen.to(device)

        scores, caps_sorted, decode_lengths, sort_ind = decoder(image_features, caps, caplens, previous_caption, prev_caplen, 
                                                                use_ss, ss_prob)


        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets  = pack_padded_sequence(targets, decode_lengths, batch_first=True)
        loss = criterion(scores.data, targets.data)
        decoder_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)
        decoder_optimizer.step()

        # Keep track of metrics
        top3 = accuracy(scores.data, targets.data, 3)
        losses.update(loss.item(), sum(decode_lengths))
        top3accs.update(top3, sum(decode_lengths))

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-3 Accuracy {top3.val:.3f} ({top3.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          loss=losses, top3=top3accs))

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
decoder_lr = 5e-4 
start_epoch = 0
epochs = 25 
epochs_since_improvement = 0 
batch_size = 80
best_cider = 0.
print_freq = 100  
checkpoint = None
annFile = 'cococaption/annotations/captions_val2014.json' 
learning_rate_decay_start = 0    
learning_rate_decay_every = 3
learning_rate_decay_rate = 0.8
use_ss = True   # wether to use scheduled sampling probability
scheduled_sampling_start = 0
scheduled_sampling_increase_every = 5
scheduled_sampling_increase_prob = 0.05
scheduled_sampling_max_prob = 0.25

# Read word map
with open('caption data/WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)
    
rev_word_map = {v: k for k, v in word_map.items()}
    
# Initialize / load checkpoint
if checkpoint is None:
    decoder = DecoderC(word_map = word_map)
    decoder_optimizer = torch.optim.Adam(params = decoder.parameters(), lr = decoder_lr)

else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_cider = checkpoint['cider']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']

# Move to GPU if available    
decoder = decoder.to(device)

# Loss functions
criterion = nn.CrossEntropyLoss().to(device)


train_loader = torch.utils.data.DataLoader(COCOTrainDataset(),
                                           batch_size = batch_size, 
                                           shuffle=True, 
                                           pin_memory=True)

val_loader = torch.utils.data.DataLoader(COCOValidationDataset(),
                                         batch_size = 1,
                                         shuffle=True, 
                                         pin_memory=True)

# Epochs
for epoch in range(start_epoch, epochs):

    if epochs_since_improvement == 3:
        print("No Improvement for 3 epochs...Early Stopping Triggered")
        break
        
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate  ** frac
        set_lr(decoder_optimizer, decoder_lr * decay_factor)
        
    if use_ss and epoch > scheduled_sampling_start:
        frac = (epoch - scheduled_sampling_start) // scheduled_sampling_increase_every
        ss_prob = min(scheduled_sampling_increase_prob  * frac, scheduled_sampling_max_prob)
    else:
        ss_prob = 0

        
    # One epoch's training
    train(train_loader=train_loader,
          decoder=decoder,
          criterion = criterion,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch, 
          word_map = word_map,
          use_ss = use_ss, 
          ss_prob = ss_prob)

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

