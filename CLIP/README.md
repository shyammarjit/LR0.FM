# CLIP

## Activate lrfm environment (Please follow [Setup.md](https://github.com/shyammarjit/LR0.FM/blob/main/MetaData/Setup.md)) & Setup CLIP locally
```
pip install -e.
```


## How to run zero shot?
```python
python resolution_zero_shot.py \
  --dataset [dataset_name] \
  --image_resolution [16|32|64|128|224] \
  --batch_size [batch_size] \
  --backbone [clip_backbone]
```
✅ Notes:
* ```dataset_name``` must be lowercase.
* ```image_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available CLIP Backbones: ```ViT-B/16```, ```ViT-L/14```, ```ViT-L/14@336px```, ```RN50```, ```RN101```, ```RN50x4```, ```RN50x16```, ```RN50x64```.

