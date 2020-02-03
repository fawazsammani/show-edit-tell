import os
import numpy as np
import h5py
import json
from tqdm import tqdm
from collections import Counter, defaultdict
from random import seed, choice, sample
import pickle


captions_per_image = 5
min_word_freq = 2 
output_folder = 'caption data'
max_len = 18


with open(os.path.join(output_folder,'dataset_coco.json'), 'r') as j:
    data = json.load(j)

with open(os.path.join('bottom-up features','train36_imgid2idx.pkl'), 'rb') as j:
    train_data = pickle.load(j)

with open(os.path.join('bottom-up features','val36_imgid2idx.pkl'), 'rb') as j:
    val_data = pickle.load(j)

# Read image paths and captions for each image
train_image_captions = []
val_image_captions = []
test_image_captions = []
train_image_det = []
val_image_det = []
test_image_det = []
train_image_names = []
val_image_names = []
test_image_names = []
word_freq = Counter()

for img in data['images']:
    captions = []
    for c in img['sentences']:
        # Update word frequency
        word_freq.update(c['tokens'])
        if len(c['tokens']) <= max_len:
            captions.append(c['tokens'])
        else:
            # clip the captions to the max length
            captions.append(c['tokens'][:max_len])

    assert len(captions) !=0

    image_id = img['filename'].split('_')[2]
    image_id = int(image_id.lstrip("0").split('.')[0])

    if img['split'] in {'train', 'restval'}:
        train_image_captions.append(captions)
        
        if img['filepath'] == 'train2014':
            assert image_id in train_data
            train_image_det.append(("t",train_data[image_id])) 
            train_image_names.append(img['filename'])
        else:
            assert image_id in val_data
            train_image_det.append(("v",val_data[image_id]))
            train_image_names.append(img['filename'])
                
    elif img['split'] in {'val'}:
        val_image_captions.append(captions)
        
        assert image_id in val_data
        val_image_det.append(("v",val_data[image_id]))
        val_image_names.append(img['filename'])
            
    elif img['split'] in {'test'}:
        test_image_captions.append(captions)
        
        assert image_id in val_data
        test_image_det.append(("v",val_data[image_id]))
        test_image_names.append(img['filename'])
                 
# Sanity check
assert len(train_image_det) == len(train_image_captions) == len(train_image_names)
assert len(val_image_det) == len(val_image_captions) == len(val_image_names)
assert len(test_image_det) == len(test_image_captions) == len(test_image_names)

# Create word map
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

# Save word map to a JSON
with open(os.path.join(output_folder, 'WORDMAP_' + 'coco' + '.json'), 'w') as j:
    json.dump(word_map, j)


for impaths, imcaps, split in [(train_image_det, train_image_captions, 'TRAIN'),
                               (val_image_det, val_image_captions, 'VAL'),
                               (test_image_det, test_image_captions, 'TEST')]:
    enc_captions = []
    caplens = []

    for i, path in enumerate(tqdm(impaths)):
        # Sample captions
        if len(imcaps[i]) < captions_per_image:
            captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
        else:
            captions = sample(imcaps[i], k=captions_per_image)

        # Sanity check
        assert len(captions) == captions_per_image

        for j, c in enumerate(captions):
            # Encode captions
            enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

            # Find caption lengths
            c_len = len(c) + 2

            enc_captions.append(enc_c)
            caplens.append(c_len)

    # Save encoded captions and their lengths to JSON files
    with open(os.path.join(output_folder, split + '_CAPTIONS_' + 'coco' + '.json'), 'w') as j:
        json.dump(enc_captions, j)

    with open(os.path.join(output_folder, split + '_CAPLENS_' + 'coco' + '.json'), 'w') as j:
        json.dump(caplens, j)

# Save bottom up features indexing to JSON files
with open(os.path.join(output_folder, 'TRAIN' + '_GENOME_DETS_' + 'coco' + '.json'), 'w') as j:
    json.dump(train_image_det, j)

with open(os.path.join(output_folder, 'VAL' + '_GENOME_DETS_' + 'coco' + '.json'), 'w') as j:
    json.dump(val_image_det, j)

with open(os.path.join(output_folder, 'TEST' + '_GENOME_DETS_' + 'coco' + '.json'), 'w') as j:
    json.dump(test_image_det, j)
    
with open(os.path.join(output_folder, 'TRAIN' + '_names_' + 'coco' + '.json'), 'w') as j:
    json.dump(train_image_names, j)

with open(os.path.join(output_folder, 'VAL' + '_names_' + 'coco' + '.json'), 'w') as j:
    json.dump(val_image_names, j)
    
with open(os.path.join(output_folder, 'TEST' + '_names_' + 'coco' + '.json'), 'w') as j:
    json.dump(test_image_names, j)

