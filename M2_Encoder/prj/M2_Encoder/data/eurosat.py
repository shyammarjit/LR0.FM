import os
import pickle
import random
import json
from glob import glob
from PIL import Image
from scipy.io import loadmat
from collections import defaultdict

from torchvision.datasets.vision import VisionDataset
class EuroSAT(VisionDataset):
    def __init__(self,
        root="", 
        train=True,
        transform=None,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.image_dir = os.path.join(self.root, "2750")
        self.class_info = os.path.join(self.root, 'split_zhou_EuroSAT.json')

        name_map = {
            'Forest': 'forest',
            'PermanentCrop': 'permanent crop land',
            'Residential': 'residential buildings or homes or apartments',
            'River': 'river',
            'Pasture': 'pasture land',
            'SeaLake': 'lake or sea',
            'HerbaceousVegetation': 'brushland or shrubland',
            'AnnualCrop': 'annual crop land',
            'Industrial': 'industrial buildings or commercial buildings',
            'Highway': 'highway or road',
        }
        
        self.image_to_class = {}
        for (i, c) in enumerate(list(name_map.keys())):
            img_files = glob(os.path.join(self.image_dir, c, "*.jpg"))
            for ik in img_files:
                split_path = ik.split("/")
                self.image_to_class[split_path[-2] + '/' + split_path[-1]] = i
        self.read_data()
        
        if self.train:
            self.images = self.train_imgs + self.val_imgs
            self.labels = self.train_labels + self.val_labels
        else:
            self.images = self.val_imgs 
            self.labels = self.val_labels

            # self.images = self.test_imgs
            # self.labels = self.test_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        target = self.labels[index]
        img = self.transform(Image.open(self.images[index]))
        return img, target
    
    def read_data(self):
        # get the train, val and test image list
        self.train_imgs, self.val_imgs, self.test_imgs = [], [], []
        with open(self.class_info, "r") as f: obj = json.load(f)
        
        self.train_imgs = [i[0] for i in obj['train']]
        self.val_imgs = [i[0] for i in obj['val']]
        self.test_imgs = [i[0] for i in obj['test']]
        self.train_labels, self.val_labels, self.test_labels = [], [], []
        for i in range(len(self.train_imgs)):
            self.train_labels.append(self.image_to_class[self.train_imgs[i]])
            self.train_imgs[i] = os.path.join(self.image_dir, self.train_imgs[i])
            if os.path.exists(self.train_imgs[i]): pass
            else: raise ValueError(f"{self.train_imgs[i]} path does not exist.")

        for i in range(len(self.val_imgs)):
            self.val_labels.append(self.image_to_class[self.val_imgs[i]])
            self.val_imgs[i] = os.path.join(self.image_dir, self.val_imgs[i])
            if os.path.exists(self.val_imgs[i]): pass
            else: raise ValueError(f"{self.val_imgs[i]} path does not exist.")

        for i in range(len(self.test_imgs)):
            self.test_labels.append(self.image_to_class[self.test_imgs[i]])
            self.test_imgs[i] = os.path.join(self.image_dir, self.test_imgs[i])
            if os.path.exists(self.test_imgs[i]): pass
            else: raise ValueError(f"{self.test_imgs[i]} path does not exist.")

        assert len(self.train_labels)==len(self.train_imgs)==13500
        assert len(self.val_labels)==len(self.val_imgs)==5400
        assert len(self.test_labels)==len(self.test_imgs)==8100