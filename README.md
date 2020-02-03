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

### Evaluation
All the generated captions and scores from our model can be found in the `outputs` folder. 

|                   | BLEU-1  | BLEU-4  |  CIDEr  |
|-------------------|:-------:|:-------:|:-------:|
|Cross-Entropy Loss |  77.9   |  38.0   |  1.200  |
|CIDEr Optimization |  80.6   |  39.2   |  1.289  |

