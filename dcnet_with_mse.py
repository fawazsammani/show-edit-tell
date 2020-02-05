import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
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
        
        # Captions per image
        self.cpi = 5

        # Load encoded captions (completely into memory)
        with open(os.path.join('caption data','TRAIN_CAPTIONS_coco.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join('caption data', 'TRAIN_CAPLENS_coco.json'), 'r') as j:
            self.caplens = json.load(j)
        
        with open('caption data/TRAIN_names_coco.json', 'r') as j:
            self.names = json.load(j)
            
        with open('caption data/CAPUTIL_train.json', 'r') as j:
            self.caption_util = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        """
        returns:
        caption: the ground-truth caption of shape (batch_size, max_length)
        caplen: the valid length (without padding) of the ground-truth caption of shape (batch_size,1)
        previous_caption: the encoded caption of the previous model of shape (batch_size, max_length)
        previous_caption_length: the valid length (without padding) of the previous caption of shape (batch_size,1)
        """
        # The Nth caption corresponds to the (N // captions_per_image)th image
        img_name = self.names[i // self.cpi]
        
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        
        previous_caption = torch.LongTensor(self.caption_util[img_name]['encoded_previous_caption'])
        prev_caplen = torch.LongTensor(self.caption_util[img_name]['previous_caption_length'])
        
        return caption, caplen, previous_caption, prev_caplen

    def __len__(self):
        return self.dataset_size
    
    
class COCOValidationDataset(Dataset):

    def __init__(self):
        
        self.cpi = 5
        
        with open('caption data/VAL_names_coco.json', 'r') as j:
            self.names = json.load(j)
            
        with open('caption data/CAPUTIL_val.json', 'r') as j:
            self.caption_util = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.names)

    def __getitem__(self, i):
        """
        returns:
        previous_caption: the encoded caption of the previous model of shape (batch_size, max_length)
        previous_caption_embed: the ELMo Max Pooled sentence embedding of the caption from the previous model of shape (batch_size, 1024)
        image_id: the respective id for the image of shape (batch_size, 1)
        previous_caption_length: the valid length (without padding) of the previous caption of shape (batch_size,1)
        """
        img_name = self.names[i]
        
        previous_caption = torch.LongTensor(self.caption_util[img_name]['encoded_previous_caption'])
        image_id = torch.LongTensor([self.caption_util[img_name]['image_ids']])
        prev_caplen = torch.LongTensor(self.caption_util[img_name]['previous_caption_length'])
        
        return image_id, previous_caption, prev_caplen

    def __len__(self):
        return self.dataset_size


def save_checkpoint(epoch, epochs_since_improvement, dae_mse, dae_mse_optimizer, cider, is_best):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'cider': cider,
             'dae_mse': dae_mse,
             'dae_mse_optimizer': dae_mse_optimizer}
    
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


def adjust_learning_rate(optimizer, shrink_factor):

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

    
def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


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
        """
        encoded captions of shape: (batch_size, max_caption_length)
        caption_lengths of shape: (batch_size, 1)
        encoded_previous_captions: encoded previous captions to be passed to the LSTM encoder of shape: (batch_size, max_caption_length)
        previous_caption_lengths of shape: (batch_size, 1)
        prev_caption_mask of shape (batch_size, max_words)
        """
        batch_size = encoded_captions.size(0)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        encoded_previous_captions = encoded_previous_captions[sort_ind]
        previous_cap_length = previous_cap_length[sort_ind]
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        decode_lengths = (caption_lengths - 1).tolist()    
        embeddings = self.embed(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device) 
        decoder_last_hidden = torch.zeros(batch_size, self.decoder_dim).to(device)
        _ , gd_final_hidden, _ = self.caption_encoder(encoded_captions, caption_lengths.unsqueeze(1))
        previous_encoded, final_hidden, prev_cap_mask = self.caption_encoder(encoded_previous_captions, previous_cap_length)
        
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
           
            topdown_input = torch.cat([embeddings[:batch_size_t, t, :],
                                       final_hidden[:batch_size_t], 
                                       h2[:batch_size_t]],dim=1)
            
            h1,c1 = self.attention_lstm(topdown_input, (h1[:batch_size_t], c1[:batch_size_t]))
            attend_cap = self.caption_attention(previous_encoded[:batch_size_t], h1[:batch_size_t], prev_cap_mask[:batch_size_t])
            
            language_input = torch.cat([h1[:batch_size_t], 
                                        attend_cap], dim = 1)

            h2,c2 = self.language_lstm(language_input, (h2[:batch_size_t], c2[:batch_size_t]))
            preds = self.fc(self.dropout(h2)) 
            predictions[:batch_size_t, t, :] = preds  
            decoder_last_hidden[:batch_size_t] = h2.clone()

        return predictions, encoded_captions, decode_lengths, sort_ind, gd_final_hidden, decoder_last_hidden

class DAEWithAR(nn.Module):
        """
        Implements DAE with MSE Optimiztion
        """
    def __init__(self):
        super(DAEWithAR, self).__init__()
        model = torch.load('BEST_checkpoint_3_dae.pth.tar')
        self.dae = model['dae']
        decoder_dim = self.dae.decoder_dim
        self.affine_hidden = nn.Linear(decoder_dim, decoder_dim)
        
    def forward(self, *args):
        scores, caps_sorted, decode_lengths, sort_ind, gd_final_hidden, decoder_last_hidden = self.dae(*args)
        decoder_last_hidden = self.affine_hidden(decoder_last_hidden)
        
        return scores, caps_sorted, decode_lengths, sort_ind, gd_final_hidden, decoder_last_hidden
     
def train(train_loader, dae_ar, criterion, mse_criterion, dae_ar_optimizer, epoch):

    dae_ar.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()  # loss (per word decoded)
    top3accs = AverageMeter()  # top5 accuracy

    for i, (caption, caplen, previous_caption, prev_caplen) in enumerate(train_loader):

        # Move to GPU, if available
        caps = caption.to(device)
        caplens = caplen.to(device)
        previous_caption = previous_caption.to(device)
        prev_caplen = prev_caplen.to(device)

        # Forward prop.
        scores, caps_sorted, decode_lengths, sort_ind, gd_final_hidden, decoder_last_hidden = dae_ar(caps, caplens, 
                                                                                                     previous_caption, 
                                                                                                     prev_caplen)
                                                  
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores.data, targets.data)
        mse_loss = mse_criterion(decoder_last_hidden, gd_final_hidden)
        
        loss += mse_loss
        
        # Back prop.
        dae_ar_optimizer.zero_grad()
        loss.backward()

        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, dae_ar.parameters()), 0.25)

        # Update weights
        dae_ar_optimizer.step()

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


def evaluate(loader, dae_ar, beam_size, epoch, word_map):
    
    vocab_size = len(word_map)
    dae_ar.eval()
    results = []
    rev_word_map = {v: k for k, v in word_map.items()}
    
    # For each image
    for i, (image_id, previous_caption, prev_caplen) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        infinite_pred = False

        # Move to GPU device, if available
        encoded_previous_captions = previous_caption.to(device) 
        prev_caplen = prev_caplen.to(device) 
        image_id = image_id.to(device)  # (1,1)
        
        previous_encoded, final_hidden, prev_caption_mask = dae_ar.dae.caption_encoder(encoded_previous_captions, prev_caplen)
        
        # Expand all
        previous_encoded = previous_encoded.expand(k, -1, -1)
        prev_cap_mask = prev_caption_mask.expand(k, -1)
        final_hidden = final_hidden.expand(k,-1)
        
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
        h1, c1 = dae_ar.dae.init_hidden_state(k)  # (batch_size, decoder_dim)
        h2, c2 = dae_ar.dae.init_hidden_state(k)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = dae_ar.dae.embed(k_prev_words).squeeze(1)        
            topdown_input = torch.cat([embeddings, final_hidden, h2],dim=1)
            h1,c1 = dae_ar.dae.attention_lstm(topdown_input, (h1, c1))
            attend_cap = dae_ar.dae.caption_attention(previous_encoded, h1, prev_cap_mask)
            language_input = torch.cat([h1, attend_cap], dim = 1)
            h2,c2 = dae_ar.dae.language_lstm(language_input, (h2, c2))
            scores = dae_ar.dae.fc(h2)  
            scores = F.log_softmax(scores, dim=1)

            # Add
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
            previous_encoded = previous_encoded[prev_word_inds[incomplete_inds]]
            prev_cap_mask = prev_cap_mask[prev_word_inds[incomplete_inds]]
            final_hidden = final_hidden[prev_word_inds[incomplete_inds]]
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
dae_lr = 5e-4 
dae_ar_lr = 1e-4 
start_epoch = 0
epochs = 10  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 60
best_cider = 0.
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none
annFile = 'cococaption/annotations/captions_val2014.json'  # Location of validation annotations
emb_file = 'glove.6B.300d.txt'

# Read word map
with open('caption data/WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)
    
rev_word_map = {v: k for k, v in word_map.items()}
    
# Initialize / load checkpoint
if checkpoint is None:
    dae_ar = DAEWithAR()
    dae_ar_optimizer = torch.optim.Adam(params=dae_ar.parameters(),lr=dae_ar_lr)

else:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_cider = checkpoint['cider']
    dae_ar = checkpoint['dae_ar']
    dae_ar_optimizer = checkpoint['dae_ar_optimizer']

dae_ar = dae_ar.to(device)

# Loss functions
criterion = nn.CrossEntropyLoss().to(device)
mse_criterion = nn.MSELoss().to(device)

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

    # terminate training if cider dosent impove for 3 consecutive epochs
    if epochs_since_improvement == 3:
        break
    
#     # Decay the learning rate by 0.8 every 3 epochs
#     if epoch % 3 == 0 and epoch !=0:
#         adjust_learning_rate(dae_optimizer, 0.8)
        
    # One epoch's training
    train(train_loader=train_loader,
          dae_ar=dae_ar,
          criterion = criterion, 
          mse_criterion = mse_criterion,
          dae_ar_optimizer=dae_ar_optimizer,
          epoch=epoch)

    # One epoch's validation
    recent_cider, recent_bleu4 = evaluate(loader = val_loader, 
                                          dae_ar = dae_ar, 
                                          beam_size = 3, 
                                          epoch = epoch, 
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
    save_checkpoint(epoch, epochs_since_improvement, dae_ar, dae_ar_optimizer, recent_cider, is_best)


