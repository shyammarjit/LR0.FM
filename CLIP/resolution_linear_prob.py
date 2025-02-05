import os
import clip
import torch
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from CLIP.dataloaders.dataset_templates import get_classes_prompts
from dataloaders import get_dataloader
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
    sub_output_folder = os.path.join(sub_output_folder, "linear-prob")
    if not os.path.exists(sub_output_folder):
        os.mkdir(sub_output_folder)

    # create the logger
    name_of_file = f"log_{args.backbone}_{args.image_resolution}"
    name_of_file = name_of_file.replace("/", "")
    setup_logger(output=sub_output_folder, name_of_file=name_of_file)
    

def main(args):
    if args.backbone in clip.available_models():
        print(f" Choosen backbone: {args.backbone}")
    else:
        raise AttributeError(f" Available clip backbone models: {clip.available_models()}")
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.backbone, device)
    if args.image_resolution!=224:
        print(f" Creating custom transformation fucntion.")
        preprocess = clip.resolution_transform(args.image_resolution)
        
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size
    
    # Preparing DATASET labels and prompts
    classes, templates = get_classes_prompts(args.dataset)

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print(f" Input image resolution {args.image_resolution}, Model resolution: {input_resolution}")
    print(f" Context length: {context_length}")
    print(f" Vocab size: {vocab_size}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    def get_features(dataset):
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
                features = model.encode_image(images.to(device))

                all_features.append(features)
                all_labels.append(labels)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

    # Load the dataset
    train, test = get_dataloader(args.dataset, preprocess, loader_type = "train")
    # loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Calculate the image features
    train_features, train_labels = get_features(train)
    test_features, test_labels = get_features(test)

    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name (small leter recombedded)",
        choices=['cifar10', 'cifar100', 'imagenet', 'stl10', 'mnist'],
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
    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()
    
    # structure the output dir
    struct_output(args)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)