## Create conda env ```lrfm```
```
conda create -y -n lrfm python=3.11
conda activate lrfm
```

## Install packages
```
python -m pip install ftfy xformers xformers packaging
python -m pip install webdataset iopath deepspeed==0.8.1
python -m pip install ptflops regex pycocoevalcap transformers==4.16.0 pytest 
pip install tqdm einops scikit-learn webdataset logger protobuf
```
