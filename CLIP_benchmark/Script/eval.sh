cd ~/LR0.FM/CLIP_benchmark/
conda activate lrfm


CLASS_DIR='~/LR0.FM/CLIP/dataloaders/classes/'
TEMPLATE_DIR='~/LR0.FM/CLIP/dataloaders/templates/'
RES=16

arr=(imagenet1k imagenet_a imagenet_v2 imagenet_r imagenet_sketch caltech101 dtd food101 sun397 cars fgvc_aircraft pets flowers eurosat ucf101) 
LOW_RES=(16 32 64 128 224)

DATASET=food101
ROOT=/data/priyank/synthetic/food-101/



python zero_shot_classification.py --backbone ViT-B-16 --pretrained datacomp_xl_s13b_b90k --image_resolution 32 --org_resolution 224 --dataset $DATASET --model_name openclip \
    --class_dir $CLASS_DIR --templates_dir $TEMPLATE_DIR --dataset_dir $ROOT 
# Top-1 accuracy: 67.27
# Top-5 accuracy: 88.82



