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

### Pretrained Models
You can download the pretrained models from [here](https://drive.google.com/drive/folders/1qyI8LD8p3qSVFC2hpVYULR8Rjr7hPBL-)

### Evaluation
All the generated captions and scores from our model can be found in the `outputs` folder. 

|                   | BLEU-1  | BLEU-4  |  CIDEr  | SPICE   |
|-------------------|:-------:|:-------:|:-------:|:-------:|
|Cross-Entropy Loss |  77.9   |  38.0   |  1.200  |  21.2   |
|CIDEr Optimization |  80.6   |  39.2   |  1.289  |  22.6   |

