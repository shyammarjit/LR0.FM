#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=10G
#SBATCH -C gmem16
#SBATCH --job-name=cars
#SBATCH --output=./outputs/cars_clip.out

source activate blip


python -m torch.distributed.run --nproc_per_node=1 train_retrieval.py \
--config ./configs/retrieval_coco_on_base.yaml \
--output_dir output/retrieval_coco_on_base \
--evaluate