# EVA-CLIP


## How to run zero shot?
```python
python resolution_zero_shot.py \
  --dataset [dataset_name] \
  --low_resolution [16|32|64|128|224] \
  --batch_size [batch_size] \
  --backbone [eva_clip_backbone]
```
✅ Notes:
* Activate the ```lrfm``` environment (Please follow [Setup.md](https://github.com/shyammarjit/LR0.FM/blob/main/MetaData/Setup.md)).
* Run the script from within the ```EVA-CLIP/rei``` directory.
* ```dataset_name``` must be lowercase.
* ```low_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available EVA-CLIP Backbones: ```EVA02-CLIP-L-14-336```, ```EVA02-CLIP-L-14```, ```EVA02-CLIP-bigE-14-plus```, ```EVA02-CLIP-bigE-14```, ```EVA02-CLIP-B-16```, ```EVA01-CLIP-g-14-plus```, ```EVA01-CLIP-g-14```.

