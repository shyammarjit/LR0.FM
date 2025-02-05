import os, random
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

def choose_resolution(n_px):
    n_px = n_px.split("/")
    if len(n_px)>1:
        # choose any random resolution
        return int(random.choice(n_px))
    else: return int(n_px[0])

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def few_shot_resolution_transform(n_px, n_px_org=224):
    """ transformation function for resolution benchmark using CLIP model. """ 
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        Resize(n_px_org, interpolation=BICUBIC),
        CenterCrop(n_px_org),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class Imagenet(Dataset):
    """
    Imagenet dataloader.
        root: str -> Path for Imagenet dataset.
        train: bool -> True for train, False for test.
        class_info: str -> Path for a *.txt file, where class directory list is there.
        transform: List -> A list of transformation function compotition.
    """
    def __init__(self, 
        root="./datasets/ImageNet", 
        train=True,
        class_info=None,
        transform=None,
        k_shot=None,
    ):
        # if train=True:
        self.root = root
        self.train = train
        self.class_info = class_info
        self.transform = transform
        self.k_shot = k_shot
        
        self.__get_categories__()

        if self.train:
            imagenet_train = os.path.join(self.root, "train")
            imagnet_path_train = [os.path.join(imagenet_train, train_dir) for train_dir in os.listdir(imagenet_train)]
            train_images = []
            for class_dir in imagnet_path_train:
                train_images.append([os.path.join(class_dir, img) for img in os.listdir(class_dir)])
            train_images = list(chain.from_iterable(train_images))
            
            assert len(imagnet_path_train)==1000 and len(train_images)==1281167
            self.images = train_images
            
        else:
            imagenet_val = os.path.join(self.root, "validation")
            imagnet_path_val = [os.path.join(imagenet_val, val_dir) for val_dir in os.listdir(imagenet_val)] 
            val_images = []
            for class_dir in imagnet_path_val:
                val_images.append([os.path.join(class_dir, img) for img in os.listdir(class_dir)])
            val_images = list(chain.from_iterable(val_images))
            
            assert len(imagnet_path_val)==1000 and len(val_images)==50000
            self.images = val_images
        
        # get the corresponding class no for each image
        self.labels = []
        for i in self.images:
            class_label = i.split('/')[-2]
            self.labels.append(int(self.dir_to_class_list[class_label]))
        
        if self.k_shot is not None: 
            if k_shot==-1:
                print("Not in few-shot settings")
            else:
                self.fewshot()
        assert len(self.labels)==len(self.images)
        self.trans = self.transform
    
    def fewshot(self):
        selected_samples, selected_labels = [], []
        unique_classes = set(self.labels)

        for class_label in unique_classes:
            class_indices = [i for i in range(len(self.images)) if self.labels[i] == class_label]
            if len(class_indices) >= self.k_shot:
                selected_indices = random.sample(class_indices, self.k_shot)
            else: selected_indices = class_indices

            selected_samples.extend([self.images[i] for i in selected_indices])
            selected_labels.extend([self.labels[i] for i in selected_indices])
        
        self.images = selected_samples
        self.labels = selected_labels

    def __len__(self):
        return len(self.images)

    def __get_categories__(self):
        if self.class_info is not None:
            # read the *.txt file
            self.dir_to_class_list = {}
            with open(self.class_info, 'r') as file:
                content = file.read(); content = str(content); content = content.split('\n', -1)
            for c in content:
                c = c.split(' ', -1)
                self.dir_to_class_list[c[0]] = c[1]
    
    def __getitem__(self, index):
        if self.k_shot is not None:
            self.trans = few_shot_resolution_transform(n_px=choose_resolution(self.transform))
            # print(self.trans)

        if self.trans is not None:
            img = self.trans(Image.open(self.images[index]))
        else:
            img = Image.open(self.images[index])
        return img, self.labels[index]