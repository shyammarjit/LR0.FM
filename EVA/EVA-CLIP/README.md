# EVA-CLIP


## How to run zero shot?
```python
python resolution_zero_shot.py \
  --dataset [dataset_name] \
  --image_resolution [16|32|64|128|224] \
  --batch_size [batch_size] \
  --backbone [eva_clip_backbone]
```
✅ Notes:
* Activate the ```lrfm``` environment.
* Run the script from within the ```rei``` directory.
* ```dataset_name``` must be lowercase.
* ```image_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available EVA-CLIP Backbones: ```2CLIPL14-336```, ```2CLIPL14```, ```2CLIPbigE14p```, ```2CLIPbigE14```, ```2CLIPB16```, ```1CLIPg14p```, ```1CLIPg14```.

