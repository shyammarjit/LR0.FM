#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=10G
#SBATCH -C gmem12
#SBATCH --job-name=fgvc_aircraft
#SBATCH --output=outputs_run/fgvc_aircraft_blip.out

source activate blip

# # backbone: ViT-B14
# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B14 \
# --resolution 384 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B14 \
# --resolution 224 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B14 \
# --resolution 128 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B14 \
# --resolution 64 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B14 \
# --resolution 32 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B14 \
# --resolution 16 \
# --batch_size 150 \
# --evaluate

# # backbone: ViT-B129
# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B129 \
# --resolution 384 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B129 \
# --resolution 224 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B129 \
# --resolution 128 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B129 \
# --resolution 64 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B129 \
# --resolution 32 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-B129 \
# --resolution 16 \
# --batch_size 150 \
# --evaluate


# # backbone: ViT-BCap129
# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-BCap129 \
# --resolution 384 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-BCap129 \
# --resolution 224 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-BCap129 \
# --resolution 128 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-BCap129 \
# --resolution 64 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-BCap129 \
# --resolution 32 \
# --batch_size 150 \
# --evaluate

# python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
# --config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
# --dataset fgvc_aircraft \
# --backbone ViT-BCap129 \
# --resolution 16 \
# --batch_size 150 \
# --evaluate


# backbone: ViT-L129
python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
--config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
--dataset fgvc_aircraft \
--backbone ViT-L129 \
--resolution 384 \
--batch_size 50 \
--evaluate

python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
--config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
--dataset fgvc_aircraft \
--backbone ViT-L129 \
--resolution 224 \
--batch_size 50 \
--evaluate

python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
--config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
--dataset fgvc_aircraft \
--backbone ViT-L129 \
--resolution 128 \
--batch_size 50 \
--evaluate

python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
--config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
--dataset fgvc_aircraft \
--backbone ViT-L129 \
--resolution 64 \
--batch_size 50 \
--evaluate

python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
--config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
--dataset fgvc_aircraft \
--backbone ViT-L129 \
--resolution 32 \
--batch_size 50 \
--evaluate

python -m torch.distributed.run --nproc_per_node=1 --master_port=35675 /home/shyam/resolution-bm/BLIP/zero_shot_classification.py \
--config /home/shyam/resolution-bm/BLIP/configs/retrieval_coco.yaml \
--dataset fgvc_aircraft \
--backbone ViT-L129 \
--resolution 16 \
--batch_size 50 \
--evaluate