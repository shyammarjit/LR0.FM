cd ~/LR0.FM/open_clip/
conda activate lrfm

NUM_GPU=1
PORT=12345
ENV='nccl'

CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'

arr=(imagenet1k imagenet_a imagenet_v2 imagenet_r imagenet_sketch caltech101 dtd food101 sun397 cars fgvc_aircraft pets flowers eurosat ucf101) 
LOW_RES=(16 32 64 128)


MODEL=OpenCLIP-ViT-B/16
NUM_TRAINING_SAMPLES=4
TRAIN_FN=train_SS_MS
OUTPUT_FILE=OC-V_B16-$TRAIN_FN
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/
RES=16
DATA=/data/priyank/Diffision_images/


############################## TRAINING 
CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
    --save-frequency 1 --zeroshot-frequency 1 --train-data $DATA --warmup 10000 \
    --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --workers=8 --epochs=30 --wd=0.1 --lr=1e-3 --batch-size=128 \
    --train-num-samples=$NUM_TRAINING_SAMPLES --seed 0 --gather-with-grad --local-loss \
    --engine $TRAIN_FN --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES \
    --lr-mode 
    

# --grad-checkpointing --precision 'amp_bf16' --beta1 0.9 --beta2 0.95 --ddp-static-graph 
# --warmup 782 --epochs=10
# --wd=0.2 --lr=2.048e-3
# --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
# --force-quick-gelu


    
    
############################## EVAL 
LR_WT=./logs/$OUTPUT_FILE/checkpoints
wt=$LR_WT"/epoch_Best.pt"
CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $dataset --low-resolution $RES --batch_size 400 --backbone $MODEL \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --lr-mode --lr-wt $wt --strict --dataset_dir $ROOT 


