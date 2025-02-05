#!/bin/bash

#SBATCH --job-name=M_7
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu

#SBATCH -C gmem48 --gres=gpu:2 --mem-per-cpu=8G -c12
#####SBATCH -C gmemT48 --gres=gpu:turing:2

#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=8G -c10
###SBATCH -C gmemT48 --gres=gpu:turing:1

###############SBATCH -p gpu --qos=day
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
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh ucf2:~/resolution-bm/EVA/EVA-CLIP/rei/ucf_output/



cd ~/resolution-bm/MetaCLIP/
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


####################################################################################
############################ META EVALS ############################


RES=16
DATASET=food101
MODEL_ARRAY=(ViT-L/14-400m ViT-L/14-2_5b ViT-B/32-400m ViT-B/32-2_5b ViT-B/16-400m ViT-B/16-2_5b ViT-H/14-2_5b ViT-bigG-14-quickgelu) 
OUTPUT_FILE=EVAL-$OUTPUT_FILE-$RES
# for MODEL in "${MODEL_ARRAY[@]}"
# do
#     printf "\n\n$MODEL \n\n" >> $OUTPUT_FILE
#     CUDA_VISIBLE_DEVICES=0 python zero_shot.py --dataset $DATASET --image_resolution $RES --batch_size 200 --backbone $MODEL \
#         --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR 
# done





####################################################################################
############################ Proposed META ############################
####################################################################################

MODEL=ViT-B/16-2_5b
TRAIN_FN=train_SS_MS
OUTPUT_FILE=B-16-$TRAIN_FN-7k-50
NAME=META-$OUTPUT_FILE
dataset=food101

LOW_RES=(16 32 64 128 224)
LOW_RES=(224)
LR_WT=./logs/$OUTPUT_FILE/checkpoints
for RES in "${LOW_RES[@]}"
do 
    OUTPUT_FILE=ucf_output/$NAME-$RES-3.txt
    printf "\n\n $NAME \n\n" >> $OUTPUT_FILE
    printf "\n\n $LR_WT \n\n" >> $OUTPUT_FILE
    wt=$LR_WT"/epoch_Best.pt"
    wt=$LR_WT"/epoch_latest.pt"
    for dataset in "${arr[@]}"
    do
        echo "$dataset"
        printf "\n\n $dataset \n\n\n$RES\n" >> $OUTPUT_FILE
        CUDA_VISIBLE_DEVICES=0 python zero_shot.py --dataset $dataset --image_resolution $RES --batch_size 200 --backbone $MODEL \
        --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --lr-mode --lr-wt $wt --strict >> $OUTPUT_FILE             
    done
done 

# MODEL=ViT-B/16-2_5b
# OUTPUT_FILE=VANILLA
# NAME=META-$OUTPUT_FILE
# LOW_RES=(16 32 64 128)
# LR_WT=./logs/$OUTPUT_FILE/checkpoints
# for RES in "${LOW_RES[@]}"
# do 
#     OUTPUT_FILE=ucf_output/$NAME-$RES.txt
#     printf "\n\n $NAME \n\n" >> $OUTPUT_FILE
#     for dataset in "${arr[@]}"
#     do
#         echo "$dataset"
#         printf "\n\n $dataset \n\n\n$RES\n" >> $OUTPUT_FILE
#         CUDA_VISIBLE_DEVICES=0 python zero_shot.py --dataset $dataset --image_resolution $RES --batch_size 200 --backbone $MODEL \
#             --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR >> $OUTPUT_FILE             
#     done
# done 

# rsync -r ucf0:~/resolution-bm/MetaCLIP/logs/ ~/resolution-bm/MetaCLIP/logs/



####################################################################################
############################ Training  ############################
####################################################################################
DATA=./datasets/Diffusion_gen_images
CONFIG='b16_fullcc'
NUM_TRAINING_SAMPLES=4
TRAIN_FN=train_SS_MS
mkdir ucf_output/
NUM_GPU=2
PORT=12346
BATCH_SIZE=128
RES=16
DATASET=food101
# # Baseline acc
# CUDA_VISIBLE_DEVICES=0 python zero_shot.py --dataset $DATASET --image_resolution 16 --batch_size 200 --backbone $MODEL \
#     --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR 


# OUTPUT_FILE=B-16-$TRAIN_FN-7k-30-EP10
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/training/main_lr.py --dist-backend $ENV \
#     --name $OUTPUT_FILE --config_name $CONFIG --zeroshot-frequency 2 --train-data $DATA --epochs=10 --workers=4 --lr-mode \
#     --pretrained='metaclip_2_5b' --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --batch-size 128 \
#     --train-num-samples=$NUM_TRAINING_SAMPLES --engine $TRAIN_FN >> ucf_output/$OUTPUT_FILE
    
    

# OUTPUT_FILE=OC-16-$TRAIN_FN-7k-30-EP10
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/training/main_lr.py --dist-backend $ENV \
#     --name $OUTPUT_FILE --config_name $CONFIG --zeroshot-frequency 2 --train-data $DATA --epochs=10 --workers=4 --lr-mode \
#     --pretrained='datacomp_xl_s13b_b90k' --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --batch-size 128 \
#     --train-num-samples=$NUM_TRAINING_SAMPLES --engine $TRAIN_FN --open-clip >> ucf_output/$OUTPUT_FILE

# INFO | Top-1 accuracy: 51.84                                                                                                                                                         
# INFO | Top-5 accuracy: 85.25     

# # args.model + "-quickgelu" & args.force_quick_gelu == True
# OUTPUT_FILE=OC-16-$TRAIN_FN-7k-30-EP10-2
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT src/training/main_lr.py --dist-backend $ENV \
#     --name $OUTPUT_FILE --config_name $CONFIG --zeroshot-frequency 2 --train-data $DATA --epochs=10 --workers=4 --lr-mode \
#     --pretrained='datacomp_xl_s13b_b90k' --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --low_resolution $RES --batch-size 128 \
#     --train-num-samples=$NUM_TRAINING_SAMPLES --engine $TRAIN_FN --open-clip >> ucf_output/$OUTPUT_FILE





rsync -a ucf_output/* ucf2:~/resolution-bm/MetaCLIP/ucf_output/

# rsync -a ucf0:~/resolution-bm/MetaCLIP/ucf_output/* ~/resolution-bm/MetaCLIP/ucf_output/


# cd ~/resolution-bm/MetaCLIP/
# sbatch Scripts/eval.sh