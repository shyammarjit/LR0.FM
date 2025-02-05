import os
import numpy as np
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset

"""
Code adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/caltech.py
Modification of caltech101 from torchvision where the background class is not removed
Thanks to the authors of torchvision
"""
from glob import glob
import os.path

from torchvision.datasets.vision import VisionDataset
class Food101(VisionDataset):
    def __init__(self,
        root="",
        transform=None,
        train=True,
    ):
        self.root = root
        self.transform = transform
        self.train = train
        
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        # read the class info from *.txt file
        with open(os.path.join(self.meta_path, 'classes.txt'), 'r') as file:
            content = file.read(); content = str(content); content = content.split('\n', -1)
            self.categories = content
            self.categories.remove("")
        assert len(self.categories)==101
            
        # read the train and test image sample info from *.txt file
        with open(os.path.join(self.meta_path, 'train.txt'), 'r') as file:
            content = file.read(); content = str(content); content = content.split('\n', -1); content.remove("")
            self.train_images = [os.path.join(self.img_path, f"{i}.jpg") for i in content] 
        assert len(self.train_images)==75750

        with open(os.path.join(self.meta_path, 'test.txt'), 'r') as file:
            content = file.read(); content = str(content); content = content.split('\n', -1); content.remove("")
            self.test_images = [os.path.join(self.img_path, f"{i}.jpg") for i in content] 
        assert len(self.test_images)==25250


        # fetch all the images and lables
        all_info = {}
        for (i, c) in enumerate(self.categories):
            imgs = glob(os.path.join(self.img_path, c, "*.jpg"))
            if len(imgs)==0: raise ValueError("No image found.")

            all_info.update(zip(imgs, len(imgs) * [i]))

        self.test_labels = [all_info[i] for i in self.test_images]
        self.train_labels = [all_info[i] for i in self.train_images]

        if self.train:
            self.images = self.train_images
            self.y = self.train_labels
        else:
            self.images = self.test_images
            self.y = self.test_labels

        assert len(self.images) == len(self.y)

    def _check_integrity(self):
        self.meta_path = os.path.join(self.root, "meta")
        self.img_path = os.path.join(self.root, "images")
        return os.path.exists(self.meta_path) and os.path.exists(self.img_path)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        target = self.y[index]
        img = self.transform(Image.open(self.images[index]))
        return img, target