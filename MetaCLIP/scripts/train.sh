
############################ LR-TK0 

cd ~/LR0.FM/MetaCLIP/
conda activate lrfm

NUM_GPU=2
PORT=12346
CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'
RES=16



DATA=/data/priyank/Diffision_images/
CONFIG='b16_fullcc'
MODEL=ViT-B/16-2_5b
NUM_TRAINING_SAMPLES=4
TRAIN_FN=train_SS_MS

DATASET=food101
VAL_DATASET=food101
VAL_ROOT=/data/priyank/synthetic/food-101/






############################ Training  
###### Vanilla Train
OUTPUT_FILE=B-16-$TRAIN_FN-7k-30-EP10
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/training/main_lr.py --dist-backend $ENV \
    --name $OUTPUT_FILE --config_name $CONFIG --zeroshot-frequency 2 --train-data $DATA --epochs=10 --workers=4 --lr-mode \
    --pretrained='metaclip_2_5b' --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --batch-size 128 \
    --train-num-samples=$NUM_TRAINING_SAMPLES --engine $TRAIN_FN --val-data $VAL_ROOT --val-datataset $VAL_DATASET
    
    
###### Open CLIP Training Config 
OUTPUT_FILE=OC-16-$TRAIN_FN-7k-30-EP10
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/training/main_lr.py --dist-backend $ENV \
    --name $OUTPUT_FILE --config_name $CONFIG --zeroshot-frequency 2 --train-data $DATA --epochs=10 --workers=4 --lr-mode \
    --pretrained='datacomp_xl_s13b_b90k' --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --batch-size 128 \
    --train-num-samples=$NUM_TRAINING_SAMPLES --engine $TRAIN_FN --val-data $VAL_ROOT --val-datataset $VAL_DATASET --open-clip 

# INFO | Top-1 accuracy: 51.84                                                                                                                                                         
# INFO | Top-5 accuracy: 85.25     




############################ EVAL   
LR_WT=./logs/FOLDER_NAME/checkpoints
wt=$LR_WT"/epoch_Best.pt"
    
CUDA_VISIBLE_DEVICES=0 python zero_shot.py --dataset $DATASET --image_resolution $RES --batch_size 200 --backbone $MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --lr-mode --lr-wt $wt --strict 
    

