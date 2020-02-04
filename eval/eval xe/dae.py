import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embedding(nn.Module):

    def __init__(self, word_map, emb_file, emb_dim, load_glove_embedding = False):
        """
        word_map: the wordmap file constructed
        emb_file: the .txt file for the glove embedding weights 
        """
        super(Embedding, self).__init__()
        
        self.emb_dim = emb_dim
        self.load_glove_embedding = load_glove_embedding
        
        if self.load_glove_embedding: 
            print("Loading GloVe...")
            with open(emb_file, 'r') as f:
                self.emb_dim = len(f.readline().split(' ')) - 1
            print("Done Loading GLoVe")
                
        self.emb_file = emb_file
        self.word_map = word_map
        self.embedding = nn.Embedding(len(word_map), self.emb_dim)  # embedding layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        if self.load_glove_embedding:
            self.load_embeddings()   
        
    def load_embeddings(self, fine_tune = True):

        vocab = set(self.word_map.keys())

        # Create tensor to hold embeddings, initialize
        embeddings = torch.FloatTensor(len(vocab), self.emb_dim)
        bias = np.sqrt(3.0 / embeddings.size(1))
        torch.nn.init.uniform_(embeddings, -bias, bias)   # initialize embeddings. Unfound words in the word_map are initialized

        # Read embedding file
        for line in open(self.emb_file, 'r', encoding="utf8"):
            line = line.split(' ')
            emb_word = line[0]
            embedded_word = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
            # Ignore word if not in vocab
            if emb_word not in vocab:
                continue   # go back and continue the loop
            embeddings[self.word_map[emb_word]] = torch.FloatTensor(embedded_word)

        self.embedding.weight = nn.Parameter(embeddings)
        
        if not fine_tune:
            for p in self.embedding.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        if self.load_glove_embedding:
            return self.embedding(x)
        else:
            out = self.embedding(x)
            out = self.relu(out)
            out = self.dropout(out)
            return out


class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hid_dim, concat_output_dim, embed):
        super(CaptionEncoder,self).__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.embed = embed
        self.lstm_encoder = nn.LSTM(emb_dim, enc_hid_dim, batch_first = True, bidirectional = True)
        self.concat = nn.Linear(enc_hid_dim * 2, concat_output_dim)

    def forward(self, src, src_len):
        """
        src: the sentence to encode of shape (batch_size, seq_length) of type Long
        src_len: long tensor that contains the lengths of each sentence in the batch of shape (batch_size, 1) of type Long
        """
        embedded = self.embed(src)  # (batch_size, seq_length, emb_dim)
        src_len = src_len.squeeze(1).tolist()
        
        packed_embedded = pack_padded_sequence(embedded, 
                                               src_len, 
                                               batch_first = True,
                                               enforce_sorted = False) # or sort then set to true (default: true)
                
        packed_outputs, hidden = self.lstm_encoder(packed_embedded)  #hidden of shape (2, batch_size, hidden_size)
        # packed sequence containing all hidden states                       
        # hidden is now from the final non-padded element in the batch
        # outputs of shape (batch_size, seq_length, hidden_size * 2)
        outputs, _ = pad_packed_sequence(packed_outputs,
                                         batch_first=True) 
        prev_cap_mask = ((outputs.sum(2))!=0).float()
        #outputs is now a non-packed sequence, all hidden states obtained when the input is a pad token are all zeros
        concat_hidden = torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim = 1)  # (batch_size, hidden_size * 2)
        final_hidden = torch.tanh(self.concat(concat_hidden))   # (batch_size, concat_output_dim)
        return outputs, final_hidden, prev_cap_mask

class CaptionAttention(nn.Module):

    def __init__(self, caption_features_dim, decoder_dim, attention_dim):

        super(CaptionAttention, self).__init__()
        self.cap_features_att = nn.Linear(caption_features_dim * 2, attention_dim) 
        self.cap_decoder_att = nn.Linear(decoder_dim, attention_dim) 
        self.cap_full_att = nn.Linear(attention_dim, 1)

    def forward(self, caption_features, decoder_hidden, prev_caption_mask):
        """
        caption features of shape: (batch_size, max_seq_length, hidden_size*2) (hidden_size = caption_features_dim)
        prev_caption_mask of shape: (batch_size, max_seq_length)
        decoder_hidden is the current output of the decoder LSTM of shape (batch_size, decoder_dim)
        text_chunk is the output of the word gating of shape (batch_size, 1024)
        """
        att1_c = self.cap_features_att(caption_features)  # (batch_size, max_words, attention_dim)
        att2_c = self.cap_decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att_c = self.cap_full_att(torch.tanh(att1_c + att2_c.unsqueeze(1))).squeeze(2)  # (batch_size, max_words)
        # Masking for zero pads for attention computation
        att_c = att_c.masked_fill(prev_caption_mask == 0, -1e10)   # (batch_size, max_words) * (batch_size, max_words)
        alpha_c = F.softmax(att_c, dim = 1)  # (batch_size, max_words)
        
        context = (caption_features * alpha_c.unsqueeze(2)).sum(dim=1)  # (batch_size, caption_features_dim)
        
        return context


class DAE(nn.Module):

    def __init__(self, 
                 word_map,  
                 emb_file,
                 decoder_dim = 1024, 
                 attention_dim = 512,
                 caption_features_dim = 512, 
                 emb_dim = 1024):
        
        super(DAE, self).__init__()
        
        self.vocab_size = len(word_map)
        self.attention_lstm = nn.LSTMCell(emb_dim * 3, decoder_dim)
        self.language_lstm = nn.LSTMCell(emb_dim * 2, decoder_dim)
        self.embed = Embedding(word_map, emb_file, emb_dim, load_glove_embedding = False)
        self.caption_encoder = CaptionEncoder(len(word_map), emb_dim, caption_features_dim, 
                                              caption_features_dim * 2, self.embed)
        self.caption_attention = CaptionAttention(caption_features_dim, decoder_dim, attention_dim)
        self.fc = nn.Linear(decoder_dim, len(word_map))
        self.tanh = nn.Tanh()
        self.decoder_dim = decoder_dim
        self.dropout = nn.Dropout(0.5)
        
    def init_hidden_state(self,batch_size):

        h = torch.zeros(batch_size,self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
        return h, c

    def forward(self, encoded_captions, caption_lengths, encoded_previous_captions, previous_cap_length):
        pass

class DAEWithAR(nn.Module):
    
    def __init__(self):
        super(DAEWithAR, self).__init__()
        
        model = torch.load('BEST_checkpoint_3_dae.pth.tar')
        self.dae = model['dae']
        decoder_dim = self.dae.decoder_dim
        self.affine_hidden = nn.Linear(decoder_dim, decoder_dim)
        
    def forward(self, *args):
        pass