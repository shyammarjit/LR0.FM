
# TRAIN LR-TK0
cd ~/LR0.FM/EVA/EVA-CLIP/rei/
conda activate pathak 
PRETRAINED_IMAGE="eva_clip"
DATA=/data/priyank/Diffision_images/
CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'
RES=16
BATCH_SIZE=64
ENV='nccl'
NUM_GPU=1



MODEL=EVA02-CLIP-bigE-14 ## 2CLIPbigE14
PRETRAINED_VISUAL_MODEL=EVA02-bigE-14

MODEL=EVA02-CLIP-L-14-336 ## 2CLIPL14-336
PRETRAINED_VISUAL_MODEL=EVA02-L-14-336   

MODEL='EVA02-CLIP-L-14' ## 2CLIPL14
PRETRAINED_VISUAL_MODEL='EVA02-L-14'

MODEL='EVA01-CLIP-g-14' ## 1CLIPg14
PRETRAINED_VISUAL_MODEL='EVA01-g-14'

MODEL='EVA01-CLIP-g-14-plus' ## 1CLIPg14p
PRETRAINED_VISUAL_MODEL='EVA01-g-14-plus'

MODEL=EVA02-CLIP-B-16 ## 2CLIPB16
PRETRAINED_VISUAL_MODEL=EVA02-B-16


PORT=12351
BATCH_SIZE=8
NUM_TRAINING_SAMPLES=4
OUTPUT_FILE=TEMP
TRAIN_FN=MS       
VAL_DATASET=food101
VAL_ROOT=/data/priyank/synthetic/food-101/




######### Training 
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT training/main_lr.py --dist-backend $ENV --name $OUTPUT_FILE \
    --save-frequency 1 --zeroshot-frequency 1 --train-num-samples 40000000 --dataset-resampled --train-data $DATA --warmup 2000 --batch-size=$BATCH_SIZE --epochs=200 --lr=5e-4 --visual-lr=2e-4 --wd=0.05 --visual-wd=0.05 --text-wd=0.05 --ld=1.0 --visual-ld=0.75 --grad-clip-norm=5.0 --smoothing=0. --workers=4 \
    --model=${MODEL} --pretrained-image=${PRETRAINED_IMAGE} --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token text_projection logit_scale --gather-with-grad --grad-checkpointing --local-loss --force-custom-clip --force-patch-dropout=0 --seed 4096 --optimizer="lamb" --zero-stage=1 --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR \
    --lr-mode --dataset-type="lr_aug" --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES --train-fn $TRAIN_FN \
    --val-dataset $VAL_DATASET  --val-data $VAL_ROOT

   	


######### Evaluation 
RES=16
MODEL=2CLIPB16
MODEL_NAME_FOR_WT=EVA02-CLIP-B-16 ## 2CLIPB16
LR_WT=./logs/FOLDER_NAME/checkpoints/epoch_Best.pt
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/

CUDA_VISIBLE_DEVICES=0 python zero_shot_classification.py --dataset $DATASET --backbone $MODEL --low_resolution $RES \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400 --lr-mode --lr-wt $wt 




