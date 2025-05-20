cd ~/LR0.FM/open_clip/
conda activate pathak
ENV='nccl'

CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'




arr=(imagenet1k imagenet_a imagenet_v2 imagenet_r imagenet_sketch caltech101 dtd food101 sun397 cars fgvc_aircraft pets flowers eurosat ucf101) 
LOW_RES=(16 32)


MODEL=OpenCLIP-ViT-B/16
DATASET=food101
ROOT=/data/priyank/synthetic/food-101/
RES=16



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
