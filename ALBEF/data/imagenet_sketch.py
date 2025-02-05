import os
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset

class Imagenet_sketch(Dataset):
    """
    Imagenet dataloader.
        root: str -> Path for Imagenet dataset.
        train: bool -> True for train, False for test.
        class_info: str -> Path for a *.txt file, where class directory list is there.
        transform: List -> A list of transformation function compotition.
    """
    def __init__(self, 
        root="", 
        train=True,
        class_info=None,
        transform=None,
    ):
        self.root = root
        self.train = train
        self.class_info = class_info
        self.transform = transform

        self.__get_categories__()

        imagnet_path_val = [os.path.join(self.root, val_dir) for val_dir in os.listdir(self.root)]
        val_images = []
        for class_dir in imagnet_path_val:
            val_images.append([os.path.join(class_dir, img) for img in os.listdir(class_dir)])
        
        val_images = list(chain.from_iterable(val_images))
        assert len(imagnet_path_val)==1000 and len(val_images)==50889
        self.images = val_images
        
        # get the corresponding class no for each image
        self.labels = []
        for i in self.images:
            class_label = i.split('/')[-2]
            self.labels.append(int(self.dir_to_class_list[class_label]))
        
        assert len(self.labels)==len(self.images)

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
        if self.transform is not None:
            img = self.transform(Image.open(self.images[index]))
        else:
            img = Image.open(self.images[index])
        return img, self.labels[index]