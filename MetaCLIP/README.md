# CLIP

## Create conda env ```metaclip```
```
conda create -y -n metaclip python=3.11
conda activate metaclip 
```

## Install packages
```
pip install -r requirements.txt
```

## Setup MetaCLIP locally
```
pip install -e.
```

## How to run zero shot?
```python
python resolution_zero_shot.py --dataset [name_of_the_dataset] --image_resolution [32] --batch_size [200] --backbone 
```
Please note that name_of_the_dataset must be in small later. Image resolution must be 16, 32, 64, 128, 224 (default).

