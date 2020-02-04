#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import json
from collections import defaultdict
from six.moves import cPickle


# In[ ]:


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    # Default value for int is 0. If no key found, the count is 0
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4):
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs

def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        # create a set for the k ref captions of one image
        for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):   # set among the 5 captions per image
            document_frequency[ngram] += 1
    return document_frequency

def build_dict(imgs, wtoi):
    
    split = 'train'
    wtoi['<end>'] = 0   
    count_imgs = 0
    refs_words = []
    refs_idxs = []
    
    for img in imgs:
        if (split == img['split']) or (split == 'train' and img['split'] == 'restval') or (split == 'all'):
            ref_words = []
            ref_idxs = []
            for sent in img['sentences']:
                tmp_tokens = sent['tokens'] + ['<end>'] 
                tmp_tokens = [_ if _ in wtoi else '<unk>' for _ in tmp_tokens]   
                ref_words.append(' '.join(tmp_tokens))
                ref_idxs.append(' '.join([str(wtoi[_]) for _ in tmp_tokens]))
            refs_words.append(ref_words)
            refs_idxs.append(ref_idxs)
            count_imgs += 1
            
    print('total imgs:', count_imgs)
    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_imgs

def pickle_dump(obj, f):
    return cPickle.dump(obj, f, protocol=2)


# In[ ]:


with open('caption data/WORDMAP_coco.json', 'r') as j:
    word_map = json.load(j)

data = json.load(open('caption data/dataset_coco.json ', 'r'))
imgs = data['images']

ngram_words, ngram_idxs, ref_len = build_dict(imgs, word_map)

pickle_dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open('data/coco-train-words.p','wb'))
pickle_dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open('data/coco-train-idxs.p','wb'))


# In[ ]:




