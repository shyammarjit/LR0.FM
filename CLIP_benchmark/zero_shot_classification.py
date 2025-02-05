"""
This script is for CLIP 
"""
import numpy as np
import os
import torch
import argparse
from tqdm import tqdm
from pkg_resources import packaging
from data import create_dataset_zero_shot
from logger import setup_logger


from clip_benchmark.metrics import zeroshot_classification
from clip_benchmark.models import MODEL_TYPES, load_clip

# print("Torch version:", torch.__version__)

def struct_output(args):
    """ create the output folder structure ."""
    # create `output' folder
    output_folder = os.path.join(args.output_dir, "output_new")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # create the dataset name as the subfolde under `output' folder  
    sub_output_folder = os.path.join(output_folder, args.dataset)
    if not os.path.exists(sub_output_folder):
        os.mkdir(sub_output_folder)
        
    # create `evaluation' type subfolder under `sub_output_folder'
    sub_output_folder = os.path.join(sub_output_folder, "zero-shot")
    if not os.path.exists(sub_output_folder):
        os.mkdir(sub_output_folder)

    
    # create the logger
    name_of_file = f"log_{args.backbone}_{args.pretrained}_{args.image_resolution}"
    name_of_file = name_of_file.replace("/", "")
    setup_logger(output=sub_output_folder, name_of_file=name_of_file)
    
def zeroshot_classifier(classnames, templates, model, tokenize):
    """ 
    Creating zero-shot classifier weights. This is taken form CLIP official codebase.
    Please refer to .
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            # print(class_embeddings.shape) # torch.Size([80, 512])
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            # print(class_embedding.shape) # torch.Size([512])
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        # print(zeroshot_weights.shape) # torch.Size([512, 1000])
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Zero-shot prediction. This is taken form CLIP official codebase.
    Please refer to .
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def read_txt(file_location):
    with open(file_location, 'r') as file:
        content = file.read(); content = str(content); content = content.split('\n', -1)
    try: content.remove("")
    except: pass
    return content

def get_classes_prompts(args):
    classes = read_txt(os.path.join(args.class_dir, f"{args.dataset}.txt"))
    templates = read_txt(os.path.join(args.templates_dir, f"{args.dataset}.txt"))
    return classes, templates


def main(args):
    # load the model.
    print(f" Model config: {args.backbone}")
    print(f" Model pretrained on {args.pretrained}")
    model, transform, tokenizer = load_clip(
        model_type='open_clip',
        model_name=args.backbone,
        pretrained=args.pretrained,
        cache_dir=None,
        device='cuda'
    )
    # print(transform)
    # exit()
    
    # Preparing DATASET labels and prompts
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print(f" Input image resolution {args.image_resolution}, Model resolution: {args.org_resolution}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    # Prepare the dataloader
    test = create_dataset_zero_shot(args.dataset, dtd_split=args.dtd_split, low_resolution=args.image_resolution, org_resolution=args.org_resolution, model_name=args.model_name, root=args.dataset_dir)
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)


    # Creating zero-shot classifier weights
    zeroshot_weights = zeroshot_classifier(classes, templates, model, tokenizer) # torch.Size([512, 1000])


    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda() # torch.Size([400, 3, 224, 224])
            target = target.cuda() # torch.Size([400])
            
            # predict
            image_features = model.encode_image(images) # torch.Size([400, 512]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
  
  

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--dataset",
        type=str,
        default="imagenet1k",
        help="Dataset name (small leter recombedded)",
        choices=['imagenet1k', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet_v2', 'caltech101', 'dtd', 'food101', 'fgvc_aircraft',\
            'sun397', 'pets', 'cars', 'flowers', 'eurosat', 'ucf101', 'birdsnap'],
    )
    parser.add_argument("--batch_size",
        type=int,
        default=400,
        help="dataloader batch size",
    )
    parser.add_argument("--backbone",
        type=str,
        default='coca_ViT-L-14',
        help="CLIP backbone model",
        # choices=['coca_ViT-L-14', 'coca_ViT-B-32', 'ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'EVA02-E-14-plus', 'ViT-SO400M-14-SigLIP-384',\
        # 'ViT-bigG-14-CLIPA-336', 'ViT-SO400M-14-SigLIP', 'ViT-bigG-14-CLIPA', 'EVA02-E-14', 'ViT-L-14-quickgelu', 'ViT-L-16-SigLIP-384',\
        # 'ViT-H-14-CLIPA-336', 'ViT-H-14-CLIPA', 'EVA02-L-14-336', 'EVA02-L-14', 'EVA01-g-14-plus', 'convnext_base', 'ViT-L-14-CLIPA-336', 'ViT-L-14-CLIPA',\
        # 'convnext_base_w', 'convnext_xxlarge', 'convnext_large_d_320', 'convnext_large_d', 'convnext_base_w_320', 'EVA01-g-14', 'EVA02-B-16'],
    )
    parser.add_argument("--pretrained",
        type=str,
        default='mscoco_finetuned_laion2b_s13b_b90k',
        help="CLIP backbone model",
        # choices=['mscoco_finetuned_laion2b_s13b_b90k', 'laion2b_s13b_b90k', 'dfn5b', 'laion2b_s9b_b144k', 'webli', 'datacomp1b', 'laion2b_s4b_b115k',\
        # 'webli', 'metaclip_fullcc', 'merged2b_s6b_b61k', 'merged2b_s11b_b114k', 'laion400m_s13b_b51k', 'laion2b', 'laion2b_s13b_b82k_augreg',\
        # 'laion_aesthetic_s13b_b82k', 'laion2b_s13b_b82k', 'laion2b_s34b_b82k_augreg_soup', 'laion2b_s34b_b82k_augreg_rewind', 'laion2b_s34b_b82k_augreg',\
        # 'laion2b_s29b_b131k_ft_soup', 'laion2b_s29b_b131k_ft', 'laion2b_s26b_b102k_augreg', 'laion_aesthetic_s13b_b82k_augreg', 'laion400m_s11b_b41k', 'merged2b_s4b_b131k',\
        # 'merged2b_s8b_b131k', ],
    )
    parser.add_argument("--num_workers",
        type=int,
        default=2,
        help="num of CPU workers in dataloader",
    )              
    parser.add_argument("--image_resolution",
        type=int,
        default=224,
        help="input image resolution for model",
    )
    parser.add_argument("--org_resolution",
        type=int,
        default=224,
        help="input image resolution for model",
    )
    parser.add_argument("--output_dir",
        type=str,
        default="./",
        help="input image resolution for model",
    )
    parser.add_argument("--dtd_split",
        type=int,
        default=1,
        help="Split number for DTD dataset, for other dataset you can ignore this.",
    )
    parser.add_argument("--class_dir",
        type=str,
        default="./CLIP/dataloaders/classes/",
        help="input image resolution for model",
    )
    parser.add_argument("--templates_dir",
        type=str,
        default="./CLIP/dataloaders/templates",
        help="input image resolution for model",
    )
    parser.add_argument("--model_name",
        type=str,
        default='siglip',
        choices=['siglip', 'coca', 'clipa', 'openclip'],
        help="model configuration name for tranformation function.",
    )

    parser.add_argument("--dataset_dir", type=str, default=None)


    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()
    
    # structure the output dir
    struct_output(args)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    