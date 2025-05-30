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
class aircraft(VisionDataset):
    def __init__(self,
        root="./datasets/fgvc-aircraft-2013b/data/",
        transform=None,
        train=True,
    ):
        # np.random.seed(1234)
        self.root = root
        self.transform = transform
        self.train = train
        
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted")
        
        # read the class info from *.txt file
        with open(os.path.join(self.root, 'variants.txt'), 'r') as file:
            content = file.read(); content = str(content); content = content.split('\n', -1)
            self.categories = content
            self.categories.remove("")
        assert len(self.categories)==100
            
        # read the train and test image sample info from *.txt file
        self.train_images, self.train_class = [], []
        self.test_images, self.test_class = [], []
        with open(os.path.join(self.root, 'images_variant_trainval.txt'), 'r') as file:
            content = file.read(); content = str(content); content = content.split('\n', -1); content.remove("")
        for i in content:
            i = i.split(" "); img_ = i[0]; class_ = " ".join(i[1:])
            self.train_images.append(os.path.join(self.img_path, f"{img_}.jpg"))
            self.train_class.append(class_)
        assert len(self.train_images)==6667

        with open(os.path.join(self.root, 'images_variant_test.txt'), 'r') as file:
            content = file.read(); content = str(content); content = content.split('\n', -1); content.remove("")
        for i in content:
            i = i.split(" "); img_ = i[0]; class_ = " ".join(i[1:])
            self.test_images.append(os.path.join(self.img_path, f"{img_}.jpg"))
            self.test_class.append(class_)
        assert len(self.test_images)==3333

        assert len(self.test_images)==len(self.test_class)
        assert len(self.train_images)==len(self.train_class)

        # fetch all the images and lables
        class_to_label = {}
        for (i, c) in enumerate(self.categories):
            class_to_label[c] = i
        
        self.train_labels = [class_to_label[i] for i in self.train_class]
        self.test_labels = [class_to_label[i] for i in self.test_class]

        if self.train:
            self.images = self.train_images
            self.y = self.train_labels
        else:
            self.images = self.test_images
            self.y = self.test_labels

        assert len(self.images) == len(self.y)

    def _check_integrity(self):
        self.img_path = os.path.join(self.root, "images")
        return os.path.exists(self.img_path)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        target = self.y[index]
        img = self.transform(Image.open(self.images[index]))
        return img, target