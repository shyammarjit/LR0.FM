# BLIP

## How to run zero shot?

```python
python -m torch.distributed.run --nproc_per_node=1 zero_shot_classification.py \
    --dataset [dataset_name] \
    --config [blip_config_file] \
    --resolution [16|32|64|128|224] \
    --batch_size [batch_size] \
    --evaluate
```

✅ Notes:
* Activate the ```lrfm``` environment (Please follow [Setup.md](https://github.com/shyammarjit/LR0.FM/blob/main/MetaData/Setup.md)). 
* Run the script from within the ```BLIP``` directory.
* ```dataset_name``` must be lowercase.
* ```resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available BLIP configs and backbones:
| Backbone | #Images | pretrained | finetuned | config file 
| ------------- | -------- | --------- | ------- |---------------------------------- |
| ViT-B | 14M | ✅ | ❌ | `pretrained_ViTB14.yaml`
| ViT-B | 129M  | ✅ | ❌ | `pretrained_ViTB129.yaml`
| ViT-BCapL | 129M | ✅ | ❌ | `pretrained_ViTBCap129.yaml`
| ViT-L | 129M | ✅ | ❌ | `pretrained_ViTL129.yaml`
| ViT-B | 129M+ | ✅ | ✅ (coco) | `finetuned_retrieval_coco_ViTB.yaml`
| ViT-L | 129M+ | ✅ | ✅ (coco) | `finetuned_retrieval_coco_ViTL.yaml`
| ViT-B | 129M+ | ✅ | ✅ (flickr) | `finetuned_retrieval_flickr_ViTB.yaml`
| ViT-L | 129M+ | ✅ | ✅ (flickr) | `finetuned_retrieval_flickr_ViTL.yaml`