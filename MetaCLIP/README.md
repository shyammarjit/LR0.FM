# MetaCLIP

## Create conda env ```metaclip```
```
conda create -y -n metaclip python=3.11
conda activate metaclip 
```

## Install packages
```
pip install -r requirements.txt
```

## Setup MetaCLIP locally
```
pip install -e.
```

## How to run zero shot?
```python
python resolution_zero_shot.py --dataset [name_of_the_dataset] --image_resolution [32] --batch_size [128] --backbone [metaclip_vit_backbone]
```
Please note that name_of_the_dataset must be in small later. Image resolution must be 16, 32, 64, 128, 224 (default).

Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

MetaCLIP backbones: ```ViT-B/32-400m```, ```ViT-B/16-400m```, ```ViT-L/14-400m```, ```ViT-B/32-2_5b```, ```ViT-B/16-2_5b```, ```ViT-L/14-2_5b```, ```ViT-H/14-2_5b```, ```ViT-bigG-14-quickgelu```.