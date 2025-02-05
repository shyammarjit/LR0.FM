import torch
from PIL import Image
# import open_clip
from src import open_clip_local as open_clip
# 512
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_400m')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
# 512
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_400m')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
# 768
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='metaclip_400m')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
# 512
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_2_5b')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
# 512
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_2_5b')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
# 768
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='metaclip_2_5b')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
# 1024
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='metaclip_2_5b', image_resolution=128)  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
# 1280
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG14-quickgelu')#, pretrained='metaclip_2_5b')  # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo

# image = preprocess(Image.open("./docs/CLIP.png")).unsqueeze(0)
# text = open_clip.tokenize(["a diagram", "a dog", "a cat"])

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     print(image_features.shape, text_features.shape) # torch.Size([1, 512]) torch.Size([3, 512]) # 
#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)





"""
This script is for CLIP 
"""
import numpy as np
import os
import torch
# import clip
import argparse
from tqdm import tqdm
# from pkg_resources import packaging
from dataloaders import get_dataloader


# print("Torch version:", torch.__version__)

def struct_output(args):
    from logger import setup_logger
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
    name_of_file = f"log_{args.backbone}_{args.image_resolution}"
    name_of_file = name_of_file.replace("/", "")
    setup_logger(output=sub_output_folder, name_of_file=name_of_file)
    
def zeroshot_classifier(classnames, templates, model):
    """ 
    Creating zero-shot classifier weights. This is taken form CLIP official codebase.
    Please refer to .
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = open_clip.tokenize(texts).cuda() #tokenize
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


def compute_flops(model, verbose=False, print_per_layer_stat=False, resolution =(3, 224, 224) ):
    from ptflops import get_model_complexity_info
    import re
    macs, params = get_model_complexity_info(model.visual.float(),  resolution , as_strings=True, print_per_layer_stat=print_per_layer_stat, verbose=verbose)
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))    
    quit()


# model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG14-quickgelu')#, pretrained='metaclip_2_5b') # 1280
def main(args):
    if args.backbone == 'ViT-B/32-400m':
        # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b' in this repo
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_400m', image_resolution=args.image_resolution) # 512
        print(f" Choosen backbone: ViT-B-32-quickgelu with pretrained=metaclip_400m")
    elif args.backbone == 'ViT-B/16-400m':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_400m', image_resolution=args.image_resolution) # 512
        print(f" Choosen backbone: ViT-B-16-quickgelu with pretrained=metaclip_400m")    
    elif args.backbone == 'ViT-L/14-400m':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='metaclip_400m', image_resolution=args.image_resolution) # 768
        print(f" Choosen backbone: ViT-L-14-quickgelu with pretrained=metaclip_400m")
    elif args.backbone == 'ViT-B/32-2_5b':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='metaclip_2_5b', image_resolution=args.image_resolution) # 512
        print(f" Choosen backbone: ViT-B-32-quickgelu with pretrained=metaclip_2_5b")
    elif args.backbone == 'ViT-B/16-2_5b':
        if args.lr_mode:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_2_5b', image_resolution=args.image_resolution, lr_mode=True, strict=False) # 512
        else:
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='metaclip_2_5b', image_resolution=args.image_resolution) # 512
        print(f" Choosen backbone: ViT-B-16-quickgelu with pretrained=metaclip_2_5b")
    elif args.backbone == 'ViT-L/14-2_5b':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='metaclip_2_5b', image_resolution=args.image_resolution) # 768
        print(f" Choosen backbone: ViT-L-14-quickgelu with pretrained=metaclip_2_5b")
    elif args.backbone == 'ViT-H/14-2_5b':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='metaclip_2_5b', image_resolution=args.image_resolution) # 1024
        print(f" Choosen backbone: ViT-L-14-quickgelu with pretrained=metaclip_2_5b")
    elif args.backbone == 'ViT-bigG-14-quickgelu':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14-quickgelu', pretrained='metaclip_2_5b', image_resolution=args.image_resolution) # 1024
        print(f" Choosen backbone: ViT-L-14-quickgelu with pretrained=metaclip_2_5b")
    else:
        raise AttributeError(f" Available clip backbone models: {clip.available_models()}")

    if args.lr_wt:
        checkpoint_path = args.lr_wt
        open_clip.factory.load_checkpoint(model, checkpoint_path, strict=args.strict)
        

    model = model.cuda()
    # load the model.
    # input_resolution = model.visual.input_resolution
    # context_length = model.context_length
    # vocab_size = model.vocab_size

    # Preparing DATASET labels and prompts
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # compute_flops(model)
    # print(f" Input image resolution {args.image_resolution}, Model resolution: {input_resolution}")
    # print(f" Context length: {context_length}")
    # print(f" Vocab size: {vocab_size}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    # Prepare the dataloader
    test = get_dataloader(args.dataset, preprocess, loader_type="test", dtd_split=args.dtd_split, root=args.dataset_dir)
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)


    # Creating zero-shot classifier weights
    zeroshot_weights = zeroshot_classifier(classes, templates, model) # torch.Size([512, 1000])


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
        help="CLIP backbone model",
        choices=['ViT-B/32-400m', 'ViT-B/16-400m', 'ViT-L/14-400m', 'ViT-B/32-2_5b', 'ViT-B/16-2_5b', 'ViT-L/14-2_5b', 'ViT-H/14-2_5b', 'ViT-bigG-14-quickgelu'],
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


    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()
    
    # structure the output dir
    struct_output(args)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)