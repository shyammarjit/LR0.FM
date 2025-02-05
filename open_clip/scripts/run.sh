CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/
cd ~/resolution-bm/CLIP_benchmark/
python zero_shot_classification.py --backbone ViT-B-16 --pretrained datacomp_xl_s13b_b90k --image_resolution 32 --org_resolution 224 --dataset $DATASET --model_name openclip \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT 
# Top-1 accuracy: 67.27
# Top-5 accuracy: 88.82



# scripts/run.sh
cd ~/resolution-bm/open_clip/
conda activate pathak 
CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'
RES=16
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/
MODEL="OpenCLIP-ViT-B/16"

CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $DATASET --low-resolution 224 --batch_size 400 --backbone=$MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT 
# Top-1 accuracy: 88.51
# Top-5 accuracy: 97.28

CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $DATASET --low-resolution 128 --batch_size 400 --backbone=$MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT 
# Top-1 accuracy: 87.01
# Top-5 accuracy: 96.99

CUDA_VISIBLE_DEVICES=1 python src/zero_shot.py --dataset $DATASET --low-resolution 64 --batch_size 400 --backbone=$MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT 
# Top-1 accuracy: 81.93
# Top-5 accuracy: 95.62

CUDA_VISIBLE_DEVICES=1 python src/zero_shot.py --dataset $DATASET --low-resolution 32 --batch_size 400 --backbone=$MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT 
# Top-1 accuracy: 67.27
# Top-5 accuracy: 88.82

CUDA_VISIBLE_DEVICES=1 python src/zero_shot.py --dataset $DATASET --low-resolution 16 --batch_size 400 --backbone=$MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT 
# Top-1 accuracy: 28.70
# Top-5 accuracy: 56.20


############################## TRAINING 
cd ~/resolution-bm/open_clip/
conda activate pathak 
ENV='nccl'
if [[ "$SLURM_JOB_NODELIST" == "c1-2" ]]; then
        echo " **** USING GLOOO ***** "
        ENV='gloo'
fi
CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'
DATA=/data/priyank/Diffision_images/


NUM_GPU=1
PORT=12345
RES=16
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/

MODEL="OpenCLIP-ViT-B/16"
NUM_TRAINING_SAMPLES=4
TRAIN_FN=train_SS_MS
OUTPUT_FILE=B-16-$TRAIN_FN


CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
    --save-frequency 1 --zeroshot-frequency 1 --train-data $DATA --warmup 10000 \
    --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --workers=8 --epochs=30 --wd=0.1 --lr=1e-3 --batch-size=128 \
    --train-num-samples=$NUM_TRAINING_SAMPLES --seed 0 --gather-with-grad --local-loss \
    --engine $TRAIN_FN --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES \
    --lr-mode --debug 


############################## EVAL 
cd ~/resolution-bm/open_clip/
conda activate pathak 
ENV='nccl'
if [[ "$SLURM_JOB_NODELIST" == "c1-2" ]]; then
        echo " **** USING GLOOO ***** "
        ENV='gloo'
fi
CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'
dataset=food101
ROOT=/data/priyank/synthetic/food-101/

MODEL="OpenCLIP-ViT-B/16"
RES=16
TRAIN_FN=train_SS_MS
LOW_RES=(16 32 64 128)
OUTPUT_FILE=OC-V_B16-$TRAIN_FN
OUTPUT_FILE=OC-V_B16-$TRAIN_FN-2

NAME=$OUTPUT_FILE
LR_WT=./logs/$OUTPUT_FILE/checkpoints
RES=16 
OUTPUT_FILE=ucf_output/$NAME-$RES.txt
wt=$LR_WT"/epoch_Best.pt"


CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $dataset --low-resolution $RES --batch_size 400 --backbone $MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --lr-mode --lr-wt $wt --strict --dataset_dir $ROOT 



rsync -a ucf0:~/resolution-bm/open_clip/ucf_output/* ~/resolution-bm/open_clip/ucf_output/