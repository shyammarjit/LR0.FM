# python -m pip install ftfy xformers xformers
# python -m pip install webdataset iopath deepspeed==0.8.1
# python -m pip install ptflops


######################################## Real World Data
cd ~/resolution-bm/EVA/EVA-CLIP/rei/
conda activate pathak3 
MODEL=2CLIPB16
DATASET="real_world"
CLASS_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/templates/'
ROOT=~/resolution-bm/REAL_WORLD/
RES=224

rsync -a ~/resolution-bm/REAL_WORLD/ ucf0:~/resolution-bm/REAL_WORLD/
CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone $MODEL --low_resolution $RES  \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --num_workers 6 --batch_size 1 --dataset_dir $ROOT


# moon_1.jpeg .... ['maraca', 'bubble', 'fig', 'planetarium', 'balloon']
# deer_0.jpeg .... ['coyote', 'cougar', 'hare', 'tick', 'lynx']
# car_0.png .... ['minivan', 'snowplow', 'parking meter', 'station wagon', 'jeep']
# galaxy_0.png .... ['planetarium', 'radio telescope', 'bubble', 'dome', 'spotlight']
# moon_0.jpeg .... ['planetarium', 'television', 'radio telescope', 'plate', 'CRT monitor']
# deer_1.jpeg .... ['gazelle', 'hare', 'red wolf or maned wolf', 'common sorrel horse', 'coyote']
# duck_0.jpg .... ['albatross', 'great egret', 'spoonbill', 'duck', 'goose']
# gun_0.jpg .... ['revolver', 'shopping basket', 'purse', 'rifle', 'grocery store']
# people_0.jpg .... ['Bloodhound', 'Dobermann', 'Rhodesian Ridgeback', 'coyote', 'Vizsla']







######################################## VANILLA EVAL
cd ~/resolution-bm/EVA/EVA-CLIP/rei/
conda activate pathak 

CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'
ROOT=/data/priyank/synthetic/food-101/

CUDA_VISIBLE_DEVICES=1 python zero_shot_classification.py --dataset food101 --backbone '1CLIPg14' --low_resolution 32 --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400
# 2CLIPbigE14 #     Model parameters: 4,704,589,569
# Top-1 accuracy: 46.19 # Top-5 accuracy: 72.44
# 2CLIPbigE14p #     Model parameters: 5,044,889,089
# Top-1 accuracy: 47.00 # Top-5 accuracy: 72.46
# 2CLIPL14-336 #     Model parameters: 428,083,137
# Top-1 accuracy: 29.33 # Top-5 accuracy: 55.21
# 2CLIPL14 #     Model parameters: 427,755,457
# Top-1 accuracy: 36.00 # Top-5 accuracy: 62.67
# 2CLIPB16 #     Model parameters: 149,691,137    
# Top-1 accuracy: 24.08 # Top-5 accuracy: 50.68
# 1CLIPg14p #     Model parameters: 1,366,621,569
# Top-1 accuracy: 40.72 # Top-5 accuracy: 69.20
# 1CLIPg14 #     Model parameters: 1,136,435,841 
# Top-1 accuracy: 26.12 # Top-5 accuracy: 52.04

 
MODEL_NAME_FOR_WT=EVA02-CLIP-B-16 ## 2CLIPB16
RES=16
N_TOKEN_LAYER=8 # [8,9,10,11]
NAME=$MODEL_NAME_FOR_WT-$RES"_MS-7k-30-16,128-$N_TOKEN_LAYER"
wt=./logs/$NAME/checkpoints/epoch_Best.pt
CUDA_VISIBLE_DEVICES=0 python zero_shot_classification.py --dataset food101 --backbone '2CLIPB16' --low_resolution 32 --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400 \
    --lr-mode --lr-wt $wt --lr-tokens-layer $N_TOKEN_LAYER 



MODEL_NAME_FOR_WT=EVA02-CLIP-B-16 ## 2CLIPB16
RES=16
NAME=$MODEL_NAME_FOR_WT-$RES"_MS-7k-30-16,128-VPT"
wt=./logs/$NAME/checkpoints/epoch_Best.pt
CUDA_VISIBLE_DEVICES=0 python zero_shot_classification.py --dataset food101 --backbone '2CLIPB16' --low_resolution 32 --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400 \
    --lr-mode --lr-wt $wt --vpt --strict


MODEL_NAME_FOR_WT=EVA02-CLIP-B-16 ## 2CLIPB16
RES=16
NAME=$MODEL_NAME_FOR_WT-$RES"_ROBUST-7k-30-16,128"
wt=./logs/$NAME/checkpoints/epoch_Best.pt
CUDA_VISIBLE_DEVICES=0 python zero_shot_classification.py --dataset food101 --backbone '2CLIPB16' --low_resolution 32 --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400 \
    --lr-mode --lr-wt $wt --robust-sam --strict
    
    


######################################## TRAIN
cd ~/resolution-bm/EVA/EVA-CLIP/rei/
conda activate pathak 
PRETRAINED_IMAGE="eva_clip"
DATA=/data/priyank/Diffision_images/
CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'
RES=16
BATCH_SIZE=64
ENV='nccl'
NUM_GPU=1
# rm ~/.cache/huggingface/hub/models--QuanSun--EVA-CLIP/snapshots/11afd202f2ae80869d6cef18b1ec775e79bd8d12/EVA02_CLIP_B_psz16_s8B.pt
# MODEL=EVA02-CLIP-bigE-14 ## 2CLIPbigE14
# PRETRAINED_VISUAL_MODEL=EVA02-bigE-14
# MODEL=EVA02-CLIP-L-14-336 ## 2CLIPL14-336
# PRETRAINED_VISUAL_MODEL=EVA02-L-14-336   
# MODEL='EVA02-CLIP-L-14' ## 2CLIPL14
# PRETRAINED_VISUAL_MODEL='EVA02-L-14'
# MODEL='EVA01-CLIP-g-14' ## 1CLIPg14
# PRETRAINED_VISUAL_MODEL='EVA01-g-14'
# MODEL='EVA01-CLIP-g-14-plus' ## 1CLIPg14p
# PRETRAINED_VISUAL_MODEL='EVA01-g-14-plus'

MODEL=EVA02-CLIP-B-16 ## 2CLIPB16
PRETRAINED_VISUAL_MODEL=EVA02-B-16
PORT=12351
BATCH_SIZE=8
# BATCH_SIZE=64
NUM_TRAINING_SAMPLES=4
OUTPUT_FILE=TEMP
TRAIN_FN=WCL-MS  
TRAIN_FN=MS       

CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT training/main_lr.py --dist-backend $ENV --name $OUTPUT_FILE \
    --save-frequency 1 --zeroshot-frequency 1 --train-num-samples 40000000 --dataset-resampled --train-data $DATA --warmup 2000 --batch-size=$BATCH_SIZE --epochs=200 --lr=5e-4 --visual-lr=2e-4 --wd=0.05 --visual-wd=0.05 --text-wd=0.05 --ld=1.0 --visual-ld=0.75 --grad-clip-norm=5.0 --smoothing=0. --workers=4 \
    --model=${MODEL} --pretrained-image=${PRETRAINED_IMAGE} --pretrained-visual-model=${PRETRAINED_VISUAL_MODEL} --skip-list head.weight head.bias lm_head.weight lm_head.bias mask_token text_projection logit_scale --gather-with-grad --grad-checkpointing --local-loss --force-custom-clip --force-patch-dropout=0 --seed 4096 --optimizer="lamb" --zero-stage=1 --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR \
    --lr-mode --dataset-type="lr_aug" --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES --train-fn $TRAIN_FN
    # >> ucf_output/$OUTPUT_FILE.txt    
   	

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
    
   	
    
    
    
    # >> ucf_output/$OUTPUT_FILE.txt    



        
        
    
######################################## ANALYIS S
cd ~/resolution-bm/EVA/EVA-CLIP/rei/
conda activate pathak 
CLASS_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/priyank/resolution-bm/CLIP/dataloaders/templates/'
RES=16

DATASET=food101
ROOT=/data/priyank/synthetic/food-101/

DATASET=fgvc_aircraft
ROOT=/data/priyank/synthetic/fgvc-aircraft-2013b/data/

DATASET=pets
ROOT=/data/priyank/synthetic/oxford_pets

DATASET=flowers
ROOT=/data/priyank/synthetic/Flowers102/flowers-102





MODEL=2CLIPB16
RES=224
RES=128
RES=16
CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone $MODEL --low_resolution $RES  \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --num_workers 6 --batch_size 400 --dataset_dir $ROOT




MODEL_NAME_FOR_WT=EVA02-CLIP-B-16 ## 2CLIPB16
TRAIN_FN=WCL-MS
NAME=$MODEL_NAME_FOR_WT-$RES"_"$TRAIN_FN-7k-50

wt=./weights/$NAME/checkpoints/epoch_Best.pt
RES=16
RES=224
arr=(food101 fgvc_aircraft pets flowers)
arr2=(food-101/ fgvc-aircraft-2013b/data/ oxford_pets Flowers102/flowers-102)
for i in "${!arr[@]}"
do
    DATASET=${arr[i]}
    y=${arr2[i]}
    ROOT=/data/priyank/synthetic/$y
    printf '%q %q\n' "$DATASET" "$ROOT"
    CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone $MODEL --low_resolution $RES --batch_size 50 \
        --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --num_workers 6 --batch_size 400 \
        --dataset_dir $ROOT --lr-mode --lr-wt $wt --strict
    
    CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone $MODEL --low_resolution $RES --batch_size 50 \
        --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --num_workers 6 --batch_size 400 \
        --dataset_dir $ROOT
done




######################################## ANALYIS (ERROR)
# UCF Predictions / Corrections 
cd ~/resolution-bm/EVA/EVA-CLIP/rei/
conda activate pathak3 
RES=128
CLASS_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/templates/'
DATASET=pets
DATASET=imagenet1k
CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution $RES --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --num_workers 6 --batch_size 400

# Top-1 accuracy: 92.23 # Top-5 accuracy: 99.81
# Top-1 accuracy: 92.64 # Top-5 accuracy: 99.86
# Top-1 accuracy: 90.32 # Top-5 accuracy: 99.67
# Top-1 accuracy: 82.53 # Top-5 accuracy: 98.53
# Top-1 accuracy: 51.84 # Top-5 accuracy: 84.71

# Top-1 accuracy: 74.10 Top-5 accuracy: 93.95
# Top-1 accuracy: 70.80 Top-5 accuracy: 91.99
# Top-1 accuracy: 58.62 Top-5 accuracy: 82.86
# Top-1 accuracy: 28.55 Top-5 accuracy: 51.01

rsync -a ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/pred_224.csv ~/resolution-bm/EVA/EVA-CLIP/rei/
rsync -a ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/pred_128.csv ~/resolution-bm/EVA/EVA-CLIP/rei/
rsync -a ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/pred_64.csv ~/resolution-bm/EVA/EVA-CLIP/rei/
rsync -a ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/pred_32.csv ~/resolution-bm/EVA/EVA-CLIP/rei/
rsync -a ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/pred_16.csv ~/resolution-bm/EVA/EVA-CLIP/rei/






CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution $RES --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --num_workers 6 --batch_size 400



cd ~/resolution-bm/EVA/EVA-CLIP/rei/
conda activate pathak3 

CLASS_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/templates/'
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/
RES=16
MODEL_NAME_FOR_WT=EVA02-CLIP-B-16 ## 2CLIPB16
NAME=$MODEL_NAME_FOR_WT-$RES"_MS-7k-30-16,128"
wt=./logs/$NAME/checkpoints/epoch_Best.pt
CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution 16 --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400 \
    --lr-mode --lr-wt $wt 
# ***** FLOP TOTAL : 17.61565056
#  Model parameters: 155,239,003

CUDA_VISIBLE_DEVICES=0 python analysis.py --dataset $DATASET --backbone '2CLIPB16' --low_resolution 16 --batch_size 50 \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT --num_workers 6 --batch_size 400 
# ***** FLOP TOTAL : 17.61565056    
#  Model parameters: 149,691,137




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







FOLDER=6
FILE=rohit_caption_853.png
cp /data/priyank/Diffision_images/Mul_samples_50_5K/$FOLDER/$FILE ./"$FOLDER"_"$FILE"
# rsync -r ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/logs/EVA02-CLIP-B-16-16_WCL-MS-7k-50 ~/resolution-bm/EVA/EVA-CLIP/rei/weights/

rsync -a ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/ucf_output/* ~/resolution-bm/EVA/EVA-CLIP/rei/ucf_output/
# rsync -r ucf0:~/resolution-bm/EVA/EVA-CLIP/rei/Analysis/feat_dump/ ~/resolution-bm/EVA/EVA-CLIP/rei/Analysis/feat_dump/



