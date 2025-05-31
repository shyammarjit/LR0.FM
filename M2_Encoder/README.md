# M2_Encoder

## How to run zero shot?
```python
python zero_shot_classification.py \
  --dataset [dataset_name] \
  --image_resolution [16|32|64|128|224] \
  --batch_size [batch_size] \
  --encode_backbone [M2_encoder_backbone]
```
✅ Notes:
* Activate the ```lrfm``` environment (Please follow [Setup.md](https://github.com/shyammarjit/LR0.FM/blob/main/MetaData/Setup.md)).
* Run the script from within the `M2_Enocoder` directory.
* ```dataset_name``` must be lowercase.
* ```image_resolution``` must be one of: 16, 32, 64, 128, 224 (default: 224).


📚 Supported Datasets: ```caltech101```, ```dtd```, ```cars```, ```sun397```, ```eurosat```, ```flowers```, ```ucf101```, ```fgvc_aircraft```, ```food101```, ```pets```, ```imagenet1k```, ```imagenet_v2```, ```imagenet_a```, ```imagenet_sketch```, ```imagenet_r```.

🧠 Available $M^2$_Encoder Backbones: ```Encoder_0.4B```, ```Encoder_1B```, ```Encoder_10B```.

