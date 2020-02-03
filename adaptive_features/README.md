### Download Bottom-up features (Skip if you are using resnet features)

#### Convert from peteanderson80's original file
Download pre-extracted features from [link](https://github.com/peteanderson80/bottom-up-attention). You can either download adaptive one or fixed one.

For example:
```
mkdir data/bu_data; cd data/bu_data
wget https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
unzip trainval.zip

```

Then:

```bash
python script/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`. If you want to use bottom-up feature, you can just follow the following steps and replace all cocotalk with cocobu.
