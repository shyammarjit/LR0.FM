#!/bin/bash

#SBATCH --job-name=OC_20
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu

#SBATCH -C gmem48 --gres=gpu:2 --mem-per-cpu=8G -c12
#####SBATCH -C gmemT48 --gres=gpu:turing:2

#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=8G -c10
###SBATCH -C gmemT48 --gres=gpu:turing:1

###SBATCH -p gpu --qos=day
###############SBATCH -p gpu --qos=short 
############SBATCH -p gpu --qos=preempt --exclude=c1-2,c1-3,c1-7,c3-4 --mem-per-cpu=8G -A preempt 




# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 --qos preempt bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem11 --qos preempt bash 
# srun --pty --gres=gpu:1 --qos preempt bash 
# srun --pty --cpus-per-task=8 bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 bash 

nvidia-smi
source ~/.bashrc
CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh ucf_output/
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh ucf2:~/resolution-bm/open_clip/ucf_output/



cd ~/resolution-bm/open_clip/
conda activate pathak3 
CLASS_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/classes/'
TEMPLATE_DIR='/home/ppriyank/resolution-bm/CLIP/dataloaders/templates/'
RES=16

DATASET=pets
arr=(imagenet1k imagenet_a imagenet_v2 imagenet_r imagenet_sketch caltech101 dtd food101 sun397 cars fgvc_aircraft pets flowers eurosat ucf101) 
LOW_RES=(16 32)


ENV='nccl'
if [[ "$SLURM_JOB_NODELIST" == "c1-2" ]]; then
        echo " **** USING GLOOO ***** "
        ENV='gloo'
fi

# mkdir ucf_output
####################################################################################
############################ CLIPA EVALS ############################


RES=16
DATASET=food101
MODEL_ARRAY=(OpenCLIP-ViT-B/16 ) 
OUTPUT_FILE=EVAL-$OUTPUT_FILE-$RES

# for MODEL in "${MODEL_ARRAY[@]}"
# do
#     printf "\n\n$MODEL \n\n" >> $OUTPUT_FILE
#     CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $DATASET --low-resolution $RES --batch_size 400 --backbone $MODEL \
#         --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR 
# done


####################################################################################
############################ Proposed META ############################
####################################################################################

MODEL="OpenCLIP-ViT-B/16"

TRAIN_FN=train_SS_MS
LOW_RES=(16 32 64 128 224)
LOW_RES=(224)
OUTPUT_FILE=OC-V_B16-$TRAIN_FN
OUTPUT_FILE=OC-V_B16-$TRAIN_FN-2
OUTPUT_FILE=OC-V_B16-$TRAIN_FN-3
OUTPUT_FILE=OC-V_B16-$TRAIN_FN-4
OUTPUT_FILE=OC-V_B16-$TRAIN_FN-5
# OUTPUT_FILE=OC-V_B16-$TRAIN_FN-6

NAME=$OUTPUT_FILE
LR_WT=./logs/$OUTPUT_FILE/checkpoints
for RES in "${LOW_RES[@]}"
do 
    OUTPUT_FILE=ucf_output/$NAME-$RES-3.txt
    printf "\n\n $NAME \n\n" >> $OUTPUT_FILE
    printf "\n\n $LR_WT \n\n" >> $OUTPUT_FILE
    wt=$LR_WT"/epoch_Best.pt"
    for dataset in "${arr[@]}"
    do
        echo "$dataset"
        printf "\n\n $dataset \n\n\n$RES\n" >> $OUTPUT_FILE
        CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $dataset --low-resolution $RES --batch_size 400 --backbone $MODEL \
            --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --lr-mode --lr-wt $wt --strict >> $OUTPUT_FILE             
    done
done 


# OUTPUT_FILE=OC-16-train_SS_MS-7k-30-EP10
# # OUTPUT_FILE=OC-16-train_SS_MS-7k-30-EP10-2
# NAME=META-$OUTPUT_FILE
# LR_WT=~/resolution-bm/MetaCLIP/logs/$OUTPUT_FILE/checkpoints
# for RES in "${LOW_RES[@]}"
# do 
#     OUTPUT_FILE=ucf_output/$NAME-$RES.txt
#     printf "\n\n $NAME \n\n" >> $OUTPUT_FILE
#     printf "\n\n $LR_WT \n\n" >> $OUTPUT_FILE
#     wt=$LR_WT"/epoch_Best.pt"
#     for dataset in "${arr[@]}"
#     do
#         echo "$dataset"
#         printf "\n\n $dataset \n\n\n$RES\n" >> $OUTPUT_FILE
#         CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $dataset --low-resolution $RES --batch_size 400 --backbone $MODEL \
#             --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --lr-mode --lr-wt $wt --strict --meta-version >> $OUTPUT_FILE             
#     done
# done 


TRAIN_FN=VANILLA        
OUTPUT_FILE=OC-V_B16-$TRAIN_FN
NAME=$OUTPUT_FILE
LOW_RES=(16 32 64 128 224)
LOW_RES=(224)
# for RES in "${LOW_RES[@]}"
# do 
#     OUTPUT_FILE=ucf_output/$NAME-$RES.txt
#     printf "\n\n $NAME \n\n" >> $OUTPUT_FILE
#     for dataset in "${arr[@]}"
#     do
#         echo "$dataset"
#         printf "\n\n $dataset \n\n\n$RES\n" >> $OUTPUT_FILE
#         CUDA_VISIBLE_DEVICES=0 python src/zero_shot.py --dataset $dataset --low-resolution $RES --batch_size 400 --backbone $MODEL \
#             --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR >> $OUTPUT_FILE             

#     done
# done 



# rsync -a ~/resolution-bm/open_clip/ucf_output/* ucf2:~/resolution-bm/open_clip/ucf_output/
# rsync -r ucf0:~/resolution-bm/open_clip/logs/ ~/resolution-bm/open_clip/logs/



####################################################################################
############################ Training  ############################
####################################################################################
# mkdir ucf_output/
DATA=./datasets/Diffusion_gen_images
RES=16
NUM_GPU=2
PORT=12352



MODEL="OpenCLIP-ViT-B/16"
NUM_TRAINING_SAMPLES=4
TRAIN_FN=train_SS_MS
OUTPUT_FILE=OC-V_B16-$TRAIN_FN



# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
#     --save-frequency 1 --zeroshot-frequency 1 --train-data $DATA --warmup 10000 \
#     --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --workers=8 --epochs=30 --wd=0.1 --lr=1e-3 --batch-size=128 \
#     --train-num-samples=$NUM_TRAINING_SAMPLES --seed 0 --gather-with-grad --local-loss \
#     --engine $TRAIN_FN --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES \
#     --lr-mode >> ucf_output/$OUTPUT_FILE.txt
#     # ~/resolution-bm/open_clip/ucf_output/$OUTPUT_FILE.txt


# OUTPUT_FILE=OC-V_B16-$TRAIN_FN-2
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
#     --save-frequency 1 --zeroshot-frequency 1 --train-data $DATA --warmup 10000 \
#     --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --workers=8 --epochs=30 --wd=0.1 --lr=1e-3 --batch-size=128 \
#     --train-num-samples=$NUM_TRAINING_SAMPLES --seed 0 --gather-with-grad --local-loss \
#     --engine $TRAIN_FN --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES \
#     --lr-mode --grad-checkpointing --precision 'amp_bf16' --beta1 0.9 --beta2 0.95 --ddp-static-graph >> ucf_output/$OUTPUT_FILE.txt
#     # ~/resolution-bm/open_clip/ucf_output/$OUTPUT_FILE.txt


# OUTPUT_FILE=OC-V_B16-$TRAIN_FN-3
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
#     --save-frequency 1 --zeroshot-frequency 1 --train-data $DATA --warmup 782 \
#     --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --workers=8 --epochs=30 --wd=0.2 --lr=2.048e-3 --batch-size=128 \
#     --train-num-samples=$NUM_TRAINING_SAMPLES --seed 0 --gather-with-grad --local-loss \
#     --engine $TRAIN_FN --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES \
#     --aug-cfg scale='(0.4, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
#     --lr-mode --grad-checkpointing --precision 'amp_bf16' --beta1 0.9 --beta2 0.95 --ddp-static-graph >> ucf_output/$OUTPUT_FILE.txt
#     # ~/resolution-bm/open_clip/ucf_output/$OUTPUT_FILE.txt

    

# OUTPUT_FILE=OC-V_B16-$TRAIN_FN-4
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
#     --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --seed 0 \
#     --zeroshot-frequency 1 --train-data $DATA --epochs=10 --workers=4 --lr-mode --workers=8 --batch-size 128 \
#     --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES \
#     --grad-checkpointing --warmup=2000 --local-loss >> ucf_output/$OUTPUT_FILE.txt
    

# OUTPUT_FILE=OC-V_B16-$TRAIN_FN-5
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
#     --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --seed 0 \
#     --zeroshot-frequency 1 --train-data $DATA --epochs=10 --workers=4 --lr-mode --workers=8 --batch-size 128 \
#     --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES \
#     --grad-checkpointing --warmup=2000 --force-quick-gelu --local-loss >> ucf_output/$OUTPUT_FILE.txt
            
    
# OUTPUT_FILE=OC-V_B16-$TRAIN_FN-6
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/open_clip_train_local/main_lr.py --dist-backend $ENV \
#     --name $OUTPUT_FILE --pretrained='datacomp_xl_s13b_b90k' --model=$MODEL --seed 0 \
#     --zeroshot-frequency 1 --train-data $DATA --epochs=200 --workers=4 --lr-mode --workers=8 --batch-size 128 \
#     --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --train-num-samples=$NUM_TRAINING_SAMPLES \
#     --grad-checkpointing --warmup=2000 --force-quick-gelu --local-loss >> ucf_output/$OUTPUT_FILE.txt
            
    
rsync -a ucf_output/* ucf2:~/resolution-bm/open_clip/ucf_output/
# rsync -a ucf0:~/resolution-bm/open_clip/ucf_output/* ~/resolution-bm/open_clip/ucf_output/



# cd ~/resolution-bm/open_clip/
# sbatch scripts/eval.sh


