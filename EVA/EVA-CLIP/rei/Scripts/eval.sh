
cd ~/LR0.FM/EVA/EVA-CLIP/rei/
conda activate lrfm 

CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'


######################################## Standard Dataset 

DATASET=food101
DATASET=imagenet1k
DATASET=imagenet_r
DATASET=imagenet_sketch
DATASET=imagenet_v2
DATASET=caltech101
DATASET=dtd
DATASET=fgvc_aircraft
DATASET=sun397
DATASET=pets
DATASET=cars
DATASET=flowers
DATASET=eurosat
DATASET=ucf101
DATASET=imagenet_a
DATASET=birdsnap


DATASET=food101
ROOT=/data/priyank/synthetic/food-101/

RES=32


MODEL=2CLIPB16
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400
# Model parameters: 1,136,435,841
# Top-1 accuracy: 70.33
# Top-5 accuracy: 90.58


MODEL=1CLIPg14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPbigE14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPbigE14p
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPL14-336
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPL14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=1CLIPg14p
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=1CLIPg14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400









