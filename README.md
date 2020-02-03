## Show, Edit and Tell: A Framework for Editing Image Captions
This contains the source code for *Show, Edit and Tell: A Framework for Editing Image Captions*

<p align="center">
  <img width="600" height="350" src="demo.png">
</p>

### Requirements
- Python 3.6 or 3.7
- PyTorch 1.0 or higher

For evaluation, you also need:
- Java 1.8.0
- [coco-api](https://github.com/cocodataset/cocoapi)
- [cococaption python 3](https://github.com/mtanti/coco-caption)


Argument Parser is currently not supported. We will add support to it soon. 

### Pretrained Models
You can download the pretrained models from [here](https://drive.google.com/drive/folders/1qyI8LD8p3qSVFC2hpVYULR8Rjr7hPBL-)

### Download Features
In this work, we use 36 fixed [bottom-up features](https://github.com/peteanderson80/bottom-up-attention). If you wish to use the adaptive features (10-100), please refer to `adaptive_features` folder in this repository and follow the instructions. 

First, download the fixed features from [here](https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip) and unzip the file. Place the unzipped folder in `bottom-up_features` folder.  

Next type this command: 
```bash
python bottom-up_features/tsv.py
```

This command will create the following files:
<ul>
<li>An HDF5 file containing the bottom up image features for train and val splits, 36 per image for each split, in an (I, 36, 2048) tensor where I is the number of images in the split.</li>
<li>PKL files that contain training and validation image IDs mapping to index in HDF5 dataset created above.</li>
</ul>

### Evaluation
All the generated captions and scores from our model can be found in the `outputs` folder. 

|                   | BLEU-1  | BLEU-4  |  CIDEr  | SPICE   |
|-------------------|:-------:|:-------:|:-------:|:-------:|
|Cross-Entropy Loss |  77.9   |  38.0   |  1.200  |  21.2   |
|CIDEr Optimization |  80.6   |  39.2   |  1.289  |  22.6   |

