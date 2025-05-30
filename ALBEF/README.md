# ALBEF

## How to run zero shot?

1. Download the pretrained checkpoints inside `ALBEF/checkpoints` folder
```python
bash download.sh
```

2. Run this inside `ALBEF` folder
```python
python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=35582 zero_shot_classification.py --config ./configs/Retrieval_coco.yaml --output_dir ./ZS --low_resolution [16|32|64|128|224] --dataset [dataset_name] --backbone [albef_backbone] --batch_size [batch_size] --evaluate
```
✅ Notes:
* Activate the ```lrfm``` environment (Please follow [Setup.md](https://github.com/shyammarjit/LR0.FM/blob/main/MetaData/Setup.md)). 
* Run the script from within the ```ALBEF``` directory.
* ```dataset_name``` must be lowercase.
* ```image_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available ALBEF Backbones: ```flicker_finetuned```, ```coco_finetuned```, ```albef_14M```, ```albef_4M```.

