import os
import numpy as np
import json
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    Implements SCMA Mechanism
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

    def forward(self, image_features, encoded_captions, caption_lengths, encoded_previous_captions, previous_cap_length, use_ss, ss_prob):
        pass
