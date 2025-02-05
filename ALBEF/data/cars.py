import os
import pickle
import math
import random
import glob
import json
from PIL import Image
from collections import defaultdict
from torchvision.datasets.vision import VisionDataset


class Cars(VisionDataset):
    def __init__(self,
        root="",
        train=True,
        transform=None,
        class_info="./data/classes/cars.txt",
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.class_info = class_info

        self.train_dir = os.path.join(self.root, "train")
        self.test_dir = os.path.join(self.root, "test")
        # print(self.train_dir, self.test_dir)
        # get the class info with label
        self.class_to_label()
        self.class_to_label = {}
        for (i, c) in enumerate(self.categories):
            if c=="Ram C/V Cargo Van Minivan 2012":
                c = "Ram C-V Cargo Van Minivan 2012"
            self.class_to_label[c] = i
        self.read_json(os.path.join(self.root, "split_zhou_StanfordCars.json"))

        if train:
            self.images = self.trainval_imgs
            self.labels = self.trainval_labels
        else:
            self.images = self.test_imgs
            self.labels = self.test_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        target = self.labels[index]
        img = self.transform(Image.open(self.images[index]))
        return img, target

    def extract_data(self, path, data):
        # data must be a List
        imgs, labels = [], [] 
        for i in range(len(data)):
            img_name, t, class_ = data[i]
            img_name = img_name.split('/')[1]
            class_ = class_.split(' ')
            class_ = class_[1:] + [class_[0]]
            class_ = ' '.join(class_)
            if class_=="Ram C/V Cargo Van Minivan 2012":
                class_ = "Ram C-V Cargo Van Minivan 2012"
            img_path = os.path.join(path, class_, img_name)
            if os.path.exists(img_path):
                imgs.append(img_path)
            else:
                raise ValueError(f"{img_path} does not exists.")
            labels.append(self.class_to_label[class_])
        return imgs, labels

    def read_json(self, fpath):
        """Read json file from a path."""
        with open(fpath, "r") as f:
            obj = json.load(f)
        # print(obj.keys()) # It has three keys => train, val, test
        self.trainval_imgs, self.trainval_labels = self.extract_data(self.train_dir, obj['train'] + obj['val'])
        self.test_imgs, self.test_labels = self.extract_data(self.test_dir, obj['test'])
        assert len(self.trainval_imgs)==len(self.trainval_labels)==8144
        assert len(self.test_imgs)==len(self.test_labels)==8041

    def class_to_label(self):
        self.categories = []
        with open(self.class_info, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.categories.append(line)