
cd ~/LR0.FM/EVA/EVA-CLIP/rei/
conda activate pathak 

CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'

DATASET=food101
ROOT=/data/priyank/synthetic/food-101/



MODEL=1CLIPg14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPbigE14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPbigE14p
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPL14-336
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPL14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=2CLIPB16
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=1CLIPg14p
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400

MODEL=1CLIPg14
CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution 32 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400




