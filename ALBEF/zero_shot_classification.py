import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from data import create_dataset_zero_shot
from scheduler import create_scheduler
from optim import create_optimizer
from logger import setup_logger


def struct_output(args):
    """ create the output folder structure ."""
    # create `output' folder
    output_folder = os.path.join(args.output_dir, "output")
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
    name_of_file = f"log_{args.backbone}_{args.low_resolution}"
    name_of_file = name_of_file.replace("/", "")
    setup_logger(output=sub_output_folder, name_of_file=name_of_file)

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    
    # Creating zero-shot classifier weights
    classes, templates = get_classes_prompts(args)
    zeroshot_weights = zeroshot_classifier(classes, templates, model, device)
    
    # Extract the image features and image embedding
    image_feats = []
    image_embeds = []

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            image_feat = model.visual_encoder(images)
            image_embed = model.vision_proj(image_feat[:,0,:])
            # image_embed = F.normalize(image_embed,dim=-1)
            # print(image_embed.shape)
            target = target.cuda()
            
            # predict
            # image_features = model.encode_image(images)
            image_embed /= image_embed.norm(dim=-1, keepdim=True)
            logits = 100. * image_embed @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")




def zeroshot_classifier(classnames, templates, model, device):
    """ 
    Creating zero-shot classifier weights. This is taken form CLIP official codebase.
    Please refer to .
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            text_input = model.tokenizer(texts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device) 
            text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
            class_embeddings = model.text_proj(text_output.last_hidden_state[:,0,:])
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
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

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    
    test_dataset = create_dataset_zero_shot(args.dataset, low_resolution=args.low_resolution, root=args.dataset_dir)   
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    evaluation(model_without_ddp, test_loader, tokenizer, device, config)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--low_resolution', default=128, type=int, help = "low resolution image size")
    parser.add_argument("--dataset",
        type=str,
        default="imagenet1k",
        help="Dataset name (small leter recombedded)",
        choices=['imagenet1k', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet_v2', 'caltech101', 'dtd', 'food101', 'fgvc_aircraft',\
            'sun397', 'pets', 'cars', 'flowers', 'eurosat', 'ucf101', 'birdsnap'],
    )
    parser.add_argument('--batch_size',
        default=256,
        type=int,
        help="test batch size"
    )
    parser.add_argument("--backbone",
        type=str,
        default='14M',
        help="CLIP backbone model",
        choices=['14M', '4M', 'coco_finetuned', 'flicker_finetuned'],
    )
    parser.add_argument("--checkpoint",
        type=str
    )
    parser.add_argument("--dtd_split",
        type=int,
        default=1,
        help="Split number for DTD dataset, for other dataset you can ignore this.",
    )
    parser.add_argument("--class_dir",
        type=str,
        default="./data/classes/",
        help="input image resolution for model, please give class dir",
    )
    parser.add_argument("--templates_dir",
        type=str,
        default="./data/templates",
        help="input image resolution for model",
    )

    parser.add_argument("--dataset_dir",
        type=str,
        default=None,
        help="input image resolution for model",
    )

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    """Download the checkpoint from official page and update the path below"""
    if args.checkpoint is None:
        if args.backbone=='14M':
            args.checkpoint = ''
        elif args.backbone=='4M':
            args.checkpoint = ''
        elif args.backbone=='coco_finetuned':
            args.checkpoint = ''
        elif args.backbone=='flicker_finetuned':
            args.checkpoint = ''
        else: raise ValueError(f"wrong checkpoint type{args.backbone}.")
    struct_output(args)
    main(args, config)
