"""
This script is for CLIP 
"""
import numpy as np
import os
import torch
import clip
import argparse
from tqdm import tqdm
from pkg_resources import packaging
from dataloaders import get_dataloader
from logger import setup_logger
from sklearn.linear_model import LogisticRegression
# print("Torch version:", torch.__version__)

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
    sub_output_folder = os.path.join(sub_output_folder, "linear-probe-few-shot")
    if not os.path.exists(sub_output_folder):
        os.mkdir(sub_output_folder)

    # create the logger
    name_of_file = f"{args.backbone}_k{args.k_shot}_train_{args.train_resolution}_test_{args.test_resolution}"
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
            texts = [template.format(classname) for template in templates] # format with class
            texts = clip.tokenize(texts).cuda() # tokenize
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
    if args.backbone in clip.available_models():
        print(f" Choosen backbone: {args.backbone}")
    else:
        raise AttributeError(f" Available clip backbone models: {clip.available_models()}")
    
    # load the model.
    model, _ = clip.load(args.backbone)
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(f" Creating custom transformation fucntion.")
    print(f'Train image resolution: {args.train_resolution}')
    print(f'Test image resolution: {args.test_resolution}')
    test_preprocess = clip.resolution_transform(args.test_resolution, n_px_org=input_resolution)

    # Preparing DATASET labels and prompts
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # print(f" Input image resolution {args.image_resolution}, Model resolution: {input_resolution}")
    print(f" Context length: {context_length}")
    print(f" Vocab size: {vocab_size}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    # Prepare the dataloader
    train, test = get_dataloader(args.dataset, test_preprocess, loader_type = "train", 
        dtd_split=args.dtd_split, 
        k_shot=args.k_shot, 
        transform_train=args.train_resolution,
        train_resolution = args.train_resolution, # extra arguments
        n_px_org=input_resolution, # extra arguments
    )
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers)

    def get_features(loader):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader)):
                features = model.encode_image(images.cuda())

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
    
    # Calculate the image features
    train_features, train_labels = get_features(train_loader)
    test_features, test_labels = get_features(test_loader)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")
  
  

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--dataset",
        type=str,
        default="imagenet1k",
        help="Dataset name (small leter recombedded)",
        choices=['imagenet1k', 'caltech101', 'dtd', 'food101', 'fgvc_aircraft',\
            'sun397', 'pets', 'cars', 'flowers', 'eurosat', 'ucf101', 'birdsnap'],
    )
    parser.add_argument("--batch_size",
        type=int,
        default=400,
        help="dataloader batch size",
    )
    parser.add_argument("--backbone",
        type=str,
        default='ViT-B/32',
        help="CLIP backbone model",
        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
    )
    parser.add_argument("--num_workers",
        type=int,
        default=2,
        help="num of CPU workers in dataloader",
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
    parser.add_argument("--k_shot",
        type=int,
        default=1,
        choices=[-1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
        help="k shot in few shot settings",
    )
    parser.add_argument("--train_resolution",
        type=str,
        default='224/128/64',
        # choices=[16, 32, 64, 128],
        help="train image resolution",
    )
    parser.add_argument("--test_resolution",
        type=int,
        default=16,
        choices=[16, 32, 64, 128],
        help="test image resolution",
    )
    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()
    
    # structure the output dir
    struct_output(args)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)