
####################################################################################
############################ META EVALS ############################

cd ~/LR0.FM/MetaCLIP/
conda activate lrfm


CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'
RES=16

DATASET=pets
DATASET=imagenet1k
DATASET=imagenet_a
DATASET=imagenet_v2
DATASET=imagenet_r
DATASET=imagenet_sketch
DATASET=caltech101
DATASET=food101
DATASET=dtd
DATASET=sun397
DATASET=cars
DATASET=fgvc_aircraft
DATASET=pets
DATASET=flowers
DATASET=eurosat
DATASET=ucf101



MODEL=ViT-L/14-400m
MODEL=ViT-L/14-2_5b
MODEL=ViT-B/32-400m
MODEL=ViT-B/32-2_5b
MODEL=ViT-B/16-400m
MODEL=ViT-B/16-2_5b
MODEL=ViT-H/14-2_5b
MODEL=ViT-bigG-14-quickgelu



MODEL=ViT-L/14-400m
RES=16
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/

CUDA_VISIBLE_DEVICES=0 python zero_shot.py --dataset $DATASET --image_resolution $RES --batch_size 200 --backbone $MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT
# Top-1 accuracy: 12.18
# Top-5 accuracy: 30.48

