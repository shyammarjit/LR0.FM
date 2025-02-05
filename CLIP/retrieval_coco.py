import torch
from torchvision.datasets import CocoCaptions
import torch.utils.data as dutils
from typing import List
import clip, argparse, os
from logger import setup_logger

# Change these to path of local COCO dataset:
coco_root = ""
coco_ann_file = "./coco/annotations/captions_val2017.json"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    sub_output_folder = os.path.join(sub_output_folder, "itm")
    if not os.path.exists(sub_output_folder):
        os.mkdir(sub_output_folder)

    # create the logger
    name_of_file = f"log_{args.backbone}_{args.image_resolution}"
    name_of_file = name_of_file.replace("/", "")
    setup_logger(output=sub_output_folder, name_of_file=name_of_file)

###########################################################################################################
# load the models and transformation 
parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--dataset",
    type=str,
    default="coco",
    help="Dataset name (small leter recombedded)",
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
parser.add_argument("--backbone",
    type=str,
    default='ViT-B/32',
    help="CLIP backbone model",
    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
)
################################################################################################
args = parser.parse_args()
# structure the output dir
struct_output(args)
model, _ = clip.load(args.backbone)
input_resolution = model.visual.input_resolution
transform = clip.resolution_transform(args.image_resolution, n_px_org=input_resolution)
model.to(device).eval()
print(f"model: {args.backbone}")
print(f"dataset: {args.dataset}")
print(f"image_resolution: {args.image_resolution}")

dataset = CocoCaptions(
    root=coco_root,
    annFile=coco_ann_file,
    transform=transform,
    # Note: almost all images have 5 captions, but 12/5000 have 6, and 1/5000 has 7 - I ignore these few extra captions.
    target_transform=lambda texts: clip.tokenize(texts[:5])
)

k_vals=[1, 5, 10, 50]


# Encodes all text and images in a dataset
def encode_dataset(clip, dataset: dutils.Dataset, batch_size = 16):
    with torch.no_grad():
        # image_to_text_map[i] gives the corresponding text indices for the ith image
        #  (as there are multiple pieces of text for each image)
        image_to_text_map = []

        # text_to_image_map[i] gives the corresponding image index for the ith text
        text_to_image_map = []

        dataloader = dutils.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        image_encodings = []
        text_encodings = []

        text_index = 0
        image_index = 0

        for images, text in dataloader:
            images = images.to(device)
            text = text.to(device)

            # text has shape B x 5 x 77
            batch_size, captions_per_image, _ = text.shape

            # Update text_to_image_map and image_to_text_map for this batch
            for i in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            # B x 5 x 77 -> (B*5) x 77
            text = torch.flatten(text, start_dim=0, end_dim=1)
            
            image_encodings.append(clip.encode_image(images))
            text_encodings.append(clip.encode_text(text))

        image_encodings = torch.cat(image_encodings)
        text_encodings = torch.cat(text_encodings)
        text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
        image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

        # Normalise encodings
        image_encodings = image_encodings / image_encodings.norm(dim=-1, keepdim=True)
        text_encodings = text_encodings / text_encodings.norm(dim=-1, keepdim=True)

        return image_encodings, text_encodings, text_to_image_map, image_to_text_map


def recall_at_k(clip, dataset: dutils.Dataset, k_vals: List[int], batch_size: int):
    print("Encoding all data...")
    image_encodings, text_encodings, text_to_image_map, image_to_text_map = encode_dataset(clip, dataset, batch_size=batch_size)
 
    num_text = text_encodings.shape[0]
    num_im = image_encodings.shape[0]
    captions_per_image = image_to_text_map.shape[1]

    # text-to-image recall
    print("Text-to-image recall...")

    dist_matrix = text_encodings @ image_encodings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)


    # image-to-text recall
    print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#

    print("Done.")
    return text_to_image_recall, image_to_text_recall


t2i, i2t = recall_at_k(model, dataset, k_vals=k_vals, batch_size=16)

print("Text-to-image Recall@K")
for k, x in zip(k_vals, t2i):
    print(f" R@{k}: {100*x:.2f}%")

print("Image-to-text Recall@K")
for k, x in zip(k_vals, i2t):
    print(f" R@{k}: {100*x:.2f}%")