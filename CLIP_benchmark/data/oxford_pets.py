import os
import pickle
import math
import random
from PIL import Image
from collections import defaultdict
from torchvision.datasets.vision import VisionDataset


class OxfordPets(VisionDataset):
    def __init__(self,
        root="./datasets/oxford_pets",
        train=True,
        transform=None,
        class_info="./CLIP_benchmark/data/classes/pets.txt",
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.class_info = class_info

        self.image_dir = os.path.join(self.root, "images")
        self.anno_dir = os.path.join(self.root, "annotations")
        # self.split_path = os.path.join(self.root, "split_zhou_OxfordPets.json")
        # get the class info with label
        self.class_to_label()


        self.trainval_imgs, self.trainval_labels, self.trainval_classes = self.read_data(split_file="trainval.txt")
        self.test_imgs, self.test_labels, self.test_classes = self.read_data(split_file="test.txt")

        assert len(self.trainval_imgs)==len(self.trainval_labels)==len(self.trainval_classes)==3680
        assert len(self.test_imgs)==len(self.test_labels)==len(self.test_classes)==3669

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

    def class_to_label(self):
        self.categories = {}
        with open(self.class_info, "r") as f:
            lines = f.readlines()
            counter = 0 # class index must start with 0, not 
            for line in lines:
                line = line.strip()
                self.categories[line.replace(' ', '_')] = counter
                counter+=1

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        impaths, labels, classnames = [], [], []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, __, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                label = self.categories[breed]
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                impaths.append(impath)
                labels.append(label)
                classnames.append(breed)
        return impaths, labels, classnames