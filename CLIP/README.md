# CLIP

## Create conda env ```clip```
```
conda create -y -n clip python=3.11
conda activate clip 
```

## Install packages
```
pip install -r requirements.txt
```

## Setup CLIP locally
```
pip install -e.
```

## How to run zero shot?
```python
python resolution_zero_shot.py --dataset [name_of_the_dataset] --image_resolution [32] --batch_size [128] --backbone [clip_vit_backbone]
```
Please note that name_of_the_dataset must be in small later. Image resolution must be 16, 32, 64, 128, 224 (default).

Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

CLIP backbones: ```ViT-B/16```, ```ViT-L/14```, ```ViT-L/14@336px```, ```RN50```, ```RN101```, ```RN50x4```, ```RN50x16```, ```RN50x64```.

