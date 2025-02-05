# CLIP

## Create conda env ```clip```
```
conda create -y -n clip python=3.11
conda activate clip 
```

## Install packages
```
pip install -r requirements.txt
```

## Setup CLIP locally
```
pip install -e.
```

## How to run zero shot?
```python
python resolution_zero_shot.py --dataset [name_of_the_dataset] --image_resolution [32]
```
Please note that name_of_the_dataset must be in small later. Image resolution must be 16, 32, 64, 128, 224 (default).

## How to run linear probing?
