import os
from PIL import Image
from itertools import chain
from torch.utils.data import Dataset

from collections import defaultdict
class REAL_WORLD(Dataset):
    def __init__(self, 
        root=None, 
        train=True,
        transform=None,
    ):
        self.root = root
        self.train = train
        self.transform = transform

        val_images = []
        for img in os.listdir(self.root):
            val_images.append(os.path.join(self.root, img))
        
        self.images = val_images
        
        # get the corresponding class no for each image
        self.labels = []
        
        self.images_index =  defaultdict(int)
        print(os.listdir(self.root) )
        curr_counter = 0 
        for i in self.images:
            curr_counter +=1
            label = curr_counter 
            self.images_index[label] = i.split('/')[-1]
            # class_label = i.split('/')[-1].replace(".png", "").split("_")[0]
            # if class_label in self.dir_to_class_list:
            #     label = self.dir_to_class_list[class_label]
            # else:
            #     label = curr_counter 
            #     curr_counter += 1
            #     self.dir_to_class_list[class_label] = label
            self.labels.append(label)
        
        assert len(self.labels)==len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.transform is not None:
            img = self.transform(Image.open(self.images[index]))
        else:
            img = Image.open(self.images[index])
        return img, self.labels[index]