# CLIP Benchmark

## Activate lrfm environment & Setup clip_benchmark locally
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
  --pretrained [backbone_is_pretrained_on_which_dataset] \
  --model_name [siglip|openclip|coca|clipa]
```

✅ Notes:
* ```dataset_name```, ```model_name```, & ```pretrained``` must be lowercase.
* ```image_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available ```coca```/```siglip```/```clipa```/```openclip``` Backbones & pretrained datset names:
| Model | Backbone | pretrained |
| -------------------- | -------------------------------------|---------------------------------- |
| `coca` | coca_ViT-L-14 | mscoco_finetuned_laion2b_s13b_b90k |
| `coca` | coca_ViT-L-14 | laion2b_s13b_b90k |
| `coca` | coca_ViT-B-32 | laion2b_s13b_b90k |
| `siglip` | ViT-SO400M-14-SigLIP | webli |
| `siglip` | ViT-SO400M-14-SigLIP-384 | webli |
| `siglip` | ViT-B-16-SigLIP-256 | webli |
| `siglip` | ViT-B-16-SigLIP | webli | 
| `siglip` | ViT-B-16-SigLIP-256 | webli |
| `siglip` | ViT-B-16-SigLIP-384 | webli |
| `siglip` | ViT-B-16-SigLIP-512 | webli | 
| `siglip` | ViT-L-16-SigLIP-256 | webli |
| `siglip` | ViT-L-16-SigLIP-384 | webli |
| `clipa` | ViT-L-14-CLIPA | datacomp1b |
| `clipa` | ViT-bigG-14-CLIPA | datacomp1b |
| `clipa` | ViT-H-14-CLIPA-336 | datacomp1b |
| `clipa` | ViT-H-14-CLIPA-336 | laion2b |
| `clipa` | ViT-H-14-CLIPA | datacomp1b |
| `clipa` | ViT-L-14-CLIPA-336 | datacomp1b |
| `clipa` | ViT-bigG-14-CLIPA-336 | datacomp1b|
| `openclip` | ViT-B-16 | datacomp_xl_s13b_b90k |
| `openclip` | ViT-B-32-256 | datacomp_s34b_b86k |
| `openclip` | ViT-L-14 | datacomp_xl_s13b_b90k |
| `openclip` | ViT-L-14 | laion2b_s32b_b82k |
| `openclip` | ViT-H-14 | laion2b_s32b_b79k |
| `openclip` | ViT-H-14-quickgelu | dfn5b |
| `openclip` | ViT-H-14-378-quickgelu | dfn5b |
| `openclip` | ViT-bigG-14 | laion2b_s39b_b160k |
