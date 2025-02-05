pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

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
from eva_clip import create_model_and_transforms, get_tokenizer

# print("Torch version:", torch.__version__)

def struct_output(args):
    """ create the output folder structure. """
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
    
def zeroshot_classifier(classnames, templates, model, model_name=None):
    """ 
    Creating zero-shot classifier weights. This is taken form CLIP official codebase.
    Please refer to .
    """
    if model_name is not None: tokenizer = get_tokenizer(model_name)
    else: raise ValueError(f'wrong {model_name}')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts)#embed with text encoder
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
    if args.lr_tokens_layer != -1:
        model, _, _ = create_model_and_transforms(args.backbone, pretrained, force_custom_clip=True, lr_clip=args.lr_mode, model_mode = args.lr_tokens_layer)    
    elif args.robust_sam or args.vpt:
        if args.robust_sam: model_mode = 3
        if args.vpt: model_mode = -2
        model, _, _ = create_model_and_transforms(args.backbone, pretrained, force_custom_clip=True, lr_clip=args.lr_mode, model_mode = model_mode)    
        model.training = False
        model.eval()
    else:
        model, _, _ = create_model_and_transforms(args.backbone, pretrained, force_custom_clip=True, lr_clip=args.lr_mode,)    
    model = model.cuda()
    
    # print(args.lr_mode, args.lr_wt )
    if args.lr_mode and args.lr_wt :
        visual_checkpoint_path = args.lr_wt
        text_checkpoint_path = ''    
        checkpoint = torch.load(args.lr_wt, map_location='cpu')
        visual_incompatible_keys = model.load_state_dict(checkpoint['state_dict'], strict=args.strict)
        # visual_incompatible_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(visual_incompatible_keys)
    # Preparing DATASET labels and prompts
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # print(f" Input image resolution {args.image_resolution}, Model resolution: {input_resolution}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    # Prepare the dataloader
    
    test = create_dataset_zero_shot(args.dataset, dtd_split=args.dtd_split, low_resolution=args.low_resolution,\
        org_resolution=336 if args.backbone=='EVA02-CLIP-L-14-336' else 224, root=args.dataset_dir, 
    )
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)


    # Creating zero-shot classifier weights
    zeroshot_weights = zeroshot_classifier(classes, templates, model, args.backbone) # torch.Size([512, 1000])


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
            'sun397', 'pets', 'cars', 'flowers', 'eurosat', 'ucf101', 'birdsnap', "real_world"],
    )
    parser.add_argument("--batch_size",
        type=int,
        default=400,
        help="dataloader batch size",
    )
    parser.add_argument("--backbone",
        type=str,
        default='2CLIPL14-336',
        help="EVA-CLIP backbone model",
        choices=['2CLIPL14-336', '2CLIPL14', '2CLIPbigE14p', '2CLIPbigE14', '2CLIPB16', '1CLIPg14p', '1CLIPg14'],
    )
    parser.add_argument("--num_workers",
        type=int,
        default=2,
        help="num of CPU workers in dataloader",
    )              
    parser.add_argument("--low_resolution",
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
    
    parser.add_argument('--lr-mode', action='store_true', default=False)
    parser.add_argument('--lr-wt', type=str, default=False)
    parser.add_argument('--strict', action='store_true', default=False)
    parser.add_argument("--lr-tokens-layer", type=int, default=-1)
    
    parser.add_argument("--vpt", action='store_true', default=False, )
    parser.add_argument("--robust-sam", action='store_true', default=False, )


    parser.add_argument("--dataset_dir",
        type=str,
        default=None,
        help="input image resolution for model",
    )
    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()
    
    if args.backbone == "2CLIPL14-336":
        args.backbone = 'EVA02-CLIP-L-14-336'
    elif args.backbone == '2CLIPL14':
        args.backbone = 'EVA02-CLIP-L-14'
    elif args.backbone == '2CLIPbigE14p':
        args.backbone = 'EVA02-CLIP-bigE-14-plus'
    elif args.backbone == '2CLIPbigE14':
        args.backbone = 'EVA02-CLIP-bigE-14'
    elif args.backbone == '2CLIPB16':
        args.backbone = 'EVA02-CLIP-B-16'
    elif args.backbone == '1CLIPg14p':
        args.backbone = 'EVA01-CLIP-g-14-plus'
    elif args.backbone == '1CLIPg14':
        args.backbone = 'EVA01-CLIP-g-14'
    else:
        raise ValueError(f'Wrong backbone type: {args.backbone}')
    # structure the output dir
    struct_output(args)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)