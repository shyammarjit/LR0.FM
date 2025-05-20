
######################################## TRAIN
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


######### Image Net Sketch hardcoded as validation Set in 

CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT training/main_lr.py --dist-backend $ENV --name $OUTPUT_FILE \
    --save-frequency 1 --zeroshot-frequency 1 --train-num-samples 40000000 --dataset-resampled --train-data $DATA --warmup 2000 --batch-size=$BATCH_SIZE --epochs=200 --lr=5e-4 --visual-lr=2e-4 --wd=0.05 --visual-wd=0.05 --text-wd=0.05 --ld=1.0 --visual-ld=0.75 --grad-clip-norm=5.0 --smoothing=0. --workers=4 \
    --model=${MODEL} --pretrained-image=${PRETRAINED_IMAGE} --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token text_projection logit_scale --gather-with-grad --grad-checkpointing --local-loss --force-custom-clip --force-patch-dropout=0 --seed 4096 --optimizer="lamb" --zero-stage=1 --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR \
    --lr-mode --dataset-type="lr_aug" --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES --train-fn $TRAIN_FN
    # >> ucf_output/$OUTPUT_FILE.txt    
   	



######################################## Training 

MODEL_NAME_FOR_WT=EVA02-CLIP-B-16 ## 2CLIPB16
RES=16
N_TOKEN_LAYER=8 # [8,9,10,11]
NAME=$MODEL_NAME_FOR_WT-$RES"_MS-7k-30-16,128-$N_TOKEN_LAYER"
wt=./logs/$NAME/checkpoints/epoch_Best.pt
CUDA_VISIBLE_DEVICES=0 python zero_shot_classification.py --dataset food101 --backbone '2CLIPB16' --low_resolution 32 --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400 \
    --lr-mode --lr-wt $wt --lr-tokens-layer $N_TOKEN_LAYER 



# TRAIN_FN=ROBUST 
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT training/main_lr.py --dist-backend $ENV --name $OUTPUT_FILE \
#     --save-frequency 1 --zeroshot-frequency 1 --train-num-samples 40000000 --dataset-resampled --train-data $DATA --warmup 2000 --batch-size=$BATCH_SIZE --epochs=200 --lr=5e-4 --visual-lr=2e-4 --wd=0.05 --visual-wd=0.05 --text-wd=0.05 --ld=1.0 --visual-ld=0.75 --grad-clip-norm=5.0 --smoothing=0. --workers=4 \
#     --model=${MODEL} --pretrained-image=${PRETRAINED_IMAGE} --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token text_projection logit_scale --gather-with-grad --grad-checkpointing --local-loss --force-custom-clip --force-patch-dropout=0 --seed 4096 --optimizer="lamb" --zero-stage=1 --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR \
#     --lr-mode --dataset-type="lr_aug" --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES --train-fn $TRAIN_FN --robust-sam
#     # >> ucf_output/$OUTPUT_FILE.txt    
   	

CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT training/main_lr.py --dist-backend $ENV --name $OUTPUT_FILE \
    --save-frequency 1 --zeroshot-frequency 1 --train-num-samples 40000000 --dataset-resampled --train-data $DATA --warmup 2000 --batch-size=$BATCH_SIZE --epochs=200 --lr=5e-4 --visual-lr=2e-4 --wd=0.05 --visual-wd=0.05 --text-wd=0.05 --ld=1.0 --visual-ld=0.75 --grad-clip-norm=5.0 --smoothing=0. --workers=4 \
    --model=${MODEL} --pretrained-image=${PRETRAINED_IMAGE} --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token text_projection logit_scale --gather-with-grad --grad-checkpointing --local-loss --force-custom-clip --force-patch-dropout=0 --seed 4096 --optimizer="lamb" --zero-stage=1 --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR \
    --lr-mode --dataset-type="lr_aug" --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES --train-fn $TRAIN_FN --no-pin-memory --vpt 
    
   	
    
    
    









######################################## SR TRAINING 
# https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb#scrollTo=GnpnrLfMV2jU
cd ~/resolution-bm/EVA/EVA-CLIP/rei/
conda activate pathak 
CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'

ROOT=/data/priyank/synthetic/oxford_pets
DATASET=pets

# SR_MODEL=HAT # SR_MODELS/HAT
SR_MODEL=BSRGAN # SR_MODELS/BSRGAN
SR_MODEL=ESRGAN # SR_MODELS/Real-ESRGAN
SR_MODEL=SwinIR # SR_MODELS/SwinIR
SR_MODEL=Inf_DiT # SR_MODELS/Inf-DiT
# SR_MODEL=ADDSR # SR_MODELS/AddSR


ROOT=/data/priyank/synthetic/food-101/
DATASET=food101

DATASET=fgvc_aircraft
ROOT=/data/priyank/synthetic/fgvc-aircraft-2013b/data/

DATASET=pets
ROOT=/data/priyank/synthetic/oxford_pets

DATASET=flowers
ROOT=/data/priyank/synthetic/Flowers102/flowers-102


LOW_RES=(16 32 64 128 224)
LOW_RES=(32)
for RES in "${LOW_RES[@]}"
do 
    # OUTPUT_FILE=ucf_output/SR_BASELINE-$RES.txt
    OUTPUT_FILE=ucf_output/$SR_MODEL-SELECTIVE-ucf2-$RES.txt
    # CUDA_VISIBLE_DEVICES=1 python zero_shot_classification_SR.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution $RES --batch_size 50 \
    #     --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 10 --SR-Model $SR_MODEL --Dump
    printf "\n\n $DATASET \n\n\n$RES\n" >> $OUTPUT_FILE
    CUDA_VISIBLE_DEVICES=1 python zero_shot_classification_SR.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution $RES --batch_size 50 \
        --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 10 --SR-Model $SR_MODEL >> $OUTPUT_FILE 
done 



LOW_RES=(16 32 64 128 224)
LOW_RES=(64 128)
LOW_RES=(64)
for RES in "${LOW_RES[@]}"
do 

    SR_MODEL=IDM # SR_MODELS/IDM
    # BASICSR_JIT=True CUDA_VISIBLE_DEVICES=0 python zero_shot_classification_SR.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution $RES --batch_size 50 \
    #     --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 10 --SR-Model $SR_MODEL --Dump

    OUTPUT_FILE=ucf_output/$SR_MODEL-SELECTIVE1-ucf2-$RES.txt
    printf "\n\n $DATASET \n\n\n$RES\n" >> $OUTPUT_FILE
    BASICSR_JIT=True CUDA_VISIBLE_DEVICES=0 python zero_shot_classification_SR.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution $RES --batch_size 50 \
        --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 50 --SR-Model $SR_MODEL >> $OUTPUT_FILE 
done 






CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution $RES --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 10
# Top-1 accuracy: 51.84
# Top-5 accuracy: 84.71


# rsync -r ~/resolution-bm/SR_MODELS/ ucf0:~/resolution-bm/SR_MODELS/






