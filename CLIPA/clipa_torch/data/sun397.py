import os
import pickle
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class SUN397(VisionDataset):
    def __init__(self,
        root="./datasets/SUN397/",
        transform=None,
        train=True,
        SPLIT="01",
    ):
        self.root = root
        self.transform = transform
        self.split_path = os.path.join(self.root, "split_zhou_SUN397.json")

        classnames = []
        with open(os.path.join(self.root, "ClassName.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()[1:]  # remove /
                classnames.append(line)
            cname2lab = {c: i for i, c in enumerate(classnames)}
            self.trainval_imgs, self.trainval_labels, self.trainval_classes = self.read_data(cname2lab, f"Training_{SPLIT}.txt")
            self.test_imgs, self.test_labels, self.test_classes = self.read_data(cname2lab, f"Testing_{SPLIT}.txt")
        
        assert len(self.trainval_imgs)==len(self.trainval_labels)==len(self.trainval_classes)
        assert len(self.test_imgs)==len(self.test_labels)==len(self.test_classes)
        assert len(set(self.test_labels))==len(set(self.trainval_labels))==397 # No oc classes must be 397

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



    def read_data(self, cname2lab, text_file):
        """
        Read a *.txt file.
        """
        text_file = os.path.join(self.root, text_file)
        impaths, labels, classnames = [], [], []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.root, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)

                impaths.append(impath)
                labels.append(label)
                classnames.append(classname)

        return impaths, labels, classnames