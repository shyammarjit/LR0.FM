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

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
class Caltech101(VisionDataset):
    """
    Caltech101 dataloader.
        root: str -> Path for Caltech101 dataset.
        train: bool -> True for train, False for test.
        class_info: str -> Path for a *.txt file, where class directory list is there.
        transform: List -> A list of transformation function compotition.
    """
    def __init__(self,
        root="./datasets/caltech-101",
        transform=None,
        train=True,
        categories=None,
    ):
        np.random.seed(1234) # as here we are choosing 30 per class for test randomly
        self.transform = transform
        self.root = root
        self.categories = categories

        self._get_categories()
        print("Total no of classes:", len(self.categories))
        
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        name_map = {
            "background": "BACKGROUND_Google",
            "off-center face": "Faces",
            "centered face": "Faces_easy",
            "leopard": "Leopards",
            "motorbike": "Motorbikes",
            "airplane": "airplanes",
            "side of a car": "car_side",
            "ceiling fan": "ceiling_fan",
            "body of a cougar cat": "cougar_body",
            "face of a cougar cat": "cougar_face",
            "head of a crocodile": "crocodile_head",
            "dollar bill": "dollar_bill",
            "electric guitar": "electric_guitar",
            "head of a flamingo": "flamingo_head",
            "grand piano": "grand_piano",
            "inline skate": "inline_skate",
            "joshua tree": "joshua_tree",
            "sea horse": "sea_horse",
            "snoopy (cartoon beagle)": "snoopy",
            "soccer ball": "soccer_ball",
            "stop sign": "stop_sign",
            "water lilly": "water_lilly",
            "wild cat": "wild_cat",
            "windsor chair": "windsor_chair",
            "yin and yang symbol": "yin_yang",
        }
        
        # get the class label list
        self.train_imgs, self.test_imgs = [], []
        self.train_labels, self.test_labels = [], []
        _TRAIN_POINTS_PER_CLASS = 30
        for (i, c) in enumerate(self.categories):
            if c in list(name_map.keys()): c = name_map[c]
            if c=="BACKGROUND_Google": continue
            imgs = glob(os.path.join(self.root, "101_ObjectCategories", c, "*.jpg"))
            if len(imgs)==0: raise ValueError("No image found.")

            train_images = list(np.random.choice(imgs, _TRAIN_POINTS_PER_CLASS, replace=False))
            test_images = list(set(imgs) - set(train_images))
            self.train_imgs.extend(train_images)
            self.test_imgs.extend(test_images)

            assert (len(train_images) + len(test_images)) == len(imgs)

            self.train_labels.extend(len(train_images) * [i])
            self.test_labels.extend(len(test_images) * [i])


        if train:
            self.images = self.train_imgs
            self.y = self.train_labels
        else:
            self.images = self.test_imgs
            self.y = self.test_labels

        assert len(self.images) == len(self.y)

    def _check_integrity(self):
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def _get_categories(self):
        with open(self.categories, 'r') as file:
            content = file.read(); content = str(content); content = content.split('\n', -1)
        self.categories = content

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        target = self.y[index]
        img = self.transform(Image.open(self.images[index]))
        return img, target