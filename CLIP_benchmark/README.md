# CLIP Benchmark

## Activate lrfm environment & Setup elip_benchmark locally
```
pip install -e. --force
```

## How to run zero shot?
```python
python resolution_zero_shot.py \
  --dataset [dataset_name] \
  --image_resolution [16|32|64|128|224] \
  --batch_size [batch_size] \
  --backbone [coca/siglip/openclip/clipa model_backbone] \
  --pretrained [backbone_is_pretrained_on_which_dataset]
```
✅ Notes:
* ```dataset_name```, ```backbone```, & ```pretrained``` must be lowercase.
* ```image_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available ```CoCa``` Backbones: ```ViT-B/16```, ```ViT-L/14```, ```ViT-L/14@336px```, ```RN50```, ```RN101```, ```RN50x4```, ```RN50x16```, ```RN50x64```.

| Model | Backbone | pretrained |
| -------------------- | -------------------------------------|---------------------------------- |
| `CoCa` | coca_ViT-L-14 | mscoco_finetuned_laion2b_s13b_b90k |
| `CoCa` | coca_ViT-L-14 | laion2b_s13b_b90k |
| `CoCa` | coca_ViT-B-32 | laion2b_s13b_b90k |
| `CoCa` |  |  |
| `CoCa` |  |  |
| `SigLIP` | ViT-SO400M-14-SigLIP | webli |
| `SigLIP` | ViT-L-16-SigLIP-256 | webli |
| `SigLIP` | ViT-B-16-SigLIP-i18n-256 | webli |
| `SigLIP` | ViT-B-16-SigLIP | webli | 
| `SigLIP` | ViT-B-16-SigLIP-256 | webli |
| `CLIPA` | ViT-L-14-CLIPA | datacomp1b |
 
