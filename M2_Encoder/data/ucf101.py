import os
import pickle
import random
import json
from glob import glob
from PIL import Image
from scipy.io import loadmat
from collections import defaultdict
import warnings 
from torchvision.datasets.vision import VisionDataset
class UCF101(VisionDataset):
    def __init__(self,
        root="", 
        train=True,
        transform=None,
        class_info="./Ant-Multi-Modal-Framework/prj/M2_Encoder/data/classes/ucf101.txt",
    ):
        
        self.root = root
        self.train = train
        self.transform = transform
        self.class_info = class_info
        self.image_dir = os.path.join(self.root, "UCF-101-midframes")
        self.split = os.path.join(self.root, "split_zhou_UCF101.json")
        
        self.read_data()

        if self.train:
            self.images = self.train_imgs + self.val_imgs
            self.labels = self.train_labels + self.val_labels
        else:
            # self.images = self.val_imgs 
            # self.labels = self.val_labels

            self.images = self.test_imgs
            self.labels = self.test_labels
        
        warnings.warn('size mismatch is there.') 
        # Here imgs are of size (320, 240)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        target = self.labels[index]
        img = self.transform(Image.open(self.images[index]))
        return img, target
    
    def read_data(self):
        # get the train, val and test image list
        self.train_imgs, self.val_imgs, self.test_imgs = [], [], []
        with open(self.split, "r") as f: 
            obj = json.load(f)
        # print(len(obj['train']))
        # print(len(obj['val']))
        # print(len(obj['test']))
        # exit()
        
        self.train_imgs = [os.path.join(self.image_dir, i[0]) for i in obj['train']]
        self.train_labels = [i[1] for i in obj['train']]
        self.val_imgs = [os.path.join(self.image_dir, i[0]) for i in obj['val']]
        self.val_labels = [i[1] for i in obj['val']]
        self.test_imgs = [os.path.join(self.image_dir, i[0]) for i in obj['test']]
        self.test_labels = [i[1] for i in obj['test']]
        
        
        
        # assert len(self.train_labels)==len(self.train_imgs)==13500
        # assert len(self.val_labels)==len(self.val_imgs)==5400
        # assert len(self.test_labels)==len(self.test_imgs)==8100