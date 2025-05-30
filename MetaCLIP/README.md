# MetaCLIP

## Activate lrfm environment (Please follow [Setup.md](https://github.com/shyammarjit/LR0.FM/blob/main/MetaData/Setup.md)) & Setup MetaCLIP locally
```
pip install -e.
```

## How to run zero shot?
```python
python resolution_zero_shot.py \
  --dataset [dataset_name] \
  --image_resolution [16|32|64|128|224] \
  --batch_size [batch_size] \
  --backbone [metaclip_backbone]
```
✅ Notes:
* ```dataset_name``` must be lowercase.
* ```image_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available MetaCLIP Backbones: ```ViT-B/32-400m```, ```ViT-B/16-400m```, ```ViT-L/14-400m```, ```ViT-B/32-2_5b```, ```ViT-B/16-2_5b```, ```ViT-L/14-2_5b```, ```ViT-H/14-2_5b```, ```ViT-bigG-14-quickgelu```.