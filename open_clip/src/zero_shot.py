import torch
import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip_local import create_model_from_pretrained, create_model_and_transforms
import open_clip

import numpy as np
import os
import argparse
from tqdm import tqdm
from data import create_dataset_zero_shot
from data.logger import setup_logger


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
    name_of_file = f"log_{args.backbone}_{args.pretrained}_{args.low_resolution}"
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



def load_meta_checkpoint(model, checkpoint_path, strict=True):
    from open_clip.factory import load_state_dict, resize_pos_embed
    state_dict = load_state_dict(checkpoint_path)
    resize_pos_embed(state_dict, model)
    for key in state_dict:
        if "visual.transformer.blocks_spatial_tokens" in key:
            state_dict[key] = state_dict[key].permute(1,0,2)
            print(key)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys




# model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG14-quickgelu')#, pretrained='metaclip_2_5b') # 1280
def main(args):

    if args.backbone == "OpenCLIP-ViT-B/16":
        # model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K')
        # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K')

        # model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K')
        # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K')

        
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        if args.lr_mode:
            model, _, preprocess_val = create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k', lr_mode=True, strict=False) # 512
        else:
            model, _, transform = create_model_and_transforms('ViT-B-16', pretrained='datacomp_xl_s13b_b90k')
            
    if args.lr_wt:
        checkpoint_path = args.lr_wt
        if args.meta_version:
            load_meta_checkpoint(model, checkpoint_path, strict=args.strict)
        else: 
            open_clip.factory.load_checkpoint(model, checkpoint_path, strict=args.strict)
        

    model = model.cuda()
    
    # Preparing DATASET labels and prompts
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # compute_flops(model)
    # print(f" Input image resolution {args.image_resolution}, Model resolution: {input_resolution}")
    # print(f" Context length: {context_length}")
    # print(f" Vocab size: {vocab_size}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    # Prepare the dataloader
    # test = create_dataset_zero_shot(args.dataset, dtd_split=args.dtd_split, low_resolution=args.low_resolution,\
    #     org_resolution=336 if "336" in args.backbone else 224, root=args.dataset_dir, 
    # )
    # loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)

    test = create_dataset_zero_shot(args.dataset, dtd_split=args.dtd_split, low_resolution=args.low_resolution, org_resolution=224, model_name='openclip', root=args.dataset_dir)
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
        default='ViT-B/32-400m',
        help="backbone model",
    )
    parser.add_argument("--num_workers",
        type=int,
        default=2,
        help="num of CPU workers in dataloader",
    )              
    parser.add_argument("--low-resolution",
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
        default="./MetaCLIP/dataloaders/classes/",
        help="input image resolution for model",
    )
    parser.add_argument("--templates_dir",
        type=str,
        default="./MetaCLIP/dataloaders/templates",
        help="input image resolution for model",
    )
    parser.add_argument("--dataset_dir", type=str, default=None, help="dataset root",)

    parser.add_argument("--lr-mode", default=None, action="store_true")
    parser.add_argument("--strict",  default=None, action="store_true")
    parser.add_argument("--lr-wt", type=str, default=None, help="dataset root",)

    parser.add_argument("--pretrained", type=str, default='datacomp_xl_s13b_b90k')


    parser.add_argument("--meta-version", default=None, action="store_true")
    
    


    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()
    
    # structure the output dir
    struct_output(args)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)