import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(txtnames, datadir, class_to_idx, SPLIT = '1'):
    images = []
    labels = []
    for txtname in txtnames:
        with open(txtname, 'r') as lines:
            for line in lines:
                classname = line.split('/')[0]
                _img = os.path.join(datadir, 'images', line.strip())
                assert os.path.isfile(_img)
                images.append(_img)
                labels.append(class_to_idx[classname])

    return images, labels

class DTD(data.Dataset):
    """
    DTD dataloader.
        root: str -> Path for DTD dataset.
        train: bool -> True for train, False for test.
        class_info: str -> Path for a *.txt file, where class directory list is there.
        transform: List -> A list of transformation function compotition.
    """
    def __init__(self, 
        root='', 
        transform=None,
        train=True, 
        SPLIT='1'
    ):
        classes, class_to_idx = find_classes(os.path.join(root, 'images'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.transform = transform

        if train:
            filename = [os.path.join(root, 'labels/train' + SPLIT + '.txt'),
                        os.path.join(root, 'labels/val' + SPLIT + '.txt')]
        else:
            filename = [os.path.join(root, 'labels/test' + SPLIT + '.txt')]

        self.images, self.labels = make_dataset(filename, root, class_to_idx, SPLIT=SPLIT)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label

    def __len__(self):
        return len(self.images)