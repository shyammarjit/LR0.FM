import os
from torchvision.datasets import CIFAR10, CIFAR100, STL10, MNIST
from .imagenet import Imagenet
from .imagenet_a import ImagenetA
from .imagenet_r import ImagenetR
from .imagenet_sketch import Imagenet_sketch
from .imagenet_v2 import Imagenet_V2
from .caltech101 import Caltech101
from .dtd import DTD
from .food101 import Food101
from .aircraft import aircraft
from .sun397 import SUN397
from .oxford_pets import OxfordPets
from .cars import Cars
from .flowers import OxfordFlowers
from .eurosat import EuroSAT
from .ucf101 import UCF101
import random
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def get_dataloader(dataset_name, transform=None, loader_type = "test", transform_train=None, k_shot=None, **kwargs):
    print(f" Dataset: {dataset_name.upper()}.")
    print(f" Transformation test: {transform}")
    print(f" Transformation train: {transform_train}")
    print(f" Dataloader type: {loader_type}.")
    
    
    if dataset_name == "cifar10":
        root = os.path.expanduser("~/.cache")
        test = CIFAR10(root, download=True, train=False, transform=transform)
        if loader_type=="test":
            print(f" Test images: {len(test)}"); return test
        elif loader_type=="train":
            train = CIFAR10(root, download=True, train=True, transform=transform_train)
            print(f" Train images: {len(train)}, Test images: {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "cifar100":
        root = os.path.expanduser("~/.cache")
        test = CIFAR100(root, download=True, train=False, transform=transform)
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = CIFAR100(root, download=True, train=True, transform=transform_train)
            print(f" Train images: {len(train)}, Test images: {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "stl10":
        root = os.path.expanduser("~/.cache")
        test = STL10(root, download=True, split='test', transform=transform)
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = STL10(root, download=True, split='train', transform=transform_train)
            print(f" Train images: {len(train)}, Test images: {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "mnist":
        root = os.path.expanduser("~/.cache")
        test = MNIST(root, download=True, train=False, transform=transform)
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = MNIST(root, download=True, train=True, transform=transform_train)
            print(f" Train images: {len(train)}, Test images: {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "imagenet1k":
        test = Imagenet(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Imagenet(root="", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet.txt",
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == 'imagenet_a':
        test = ImagenetA(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_a.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = ImagenetA(root="", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_a.txt",
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == 'imagenet_r':
        test = ImagenetR(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_r.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = ImagenetR(root="", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_r.txt",
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == 'imagenet_sketch':
        test = Imagenet_sketch(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_sketch.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Imagenet_sketch(root="", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_sketch.txt",
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == 'imagenet_v2':
        test = Imagenet_V2(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_v2.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Imagenet_V2(root="", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_v2.txt",
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == "caltech101":
        test = Caltech101(root="", 
            train=False,
            transform=transform,
            categories="./CLIP/dataloaders/classes/caltech101.txt",
        )

        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Caltech101(root="", 
                train=True,
                transform=transform_train,
                categories="./CLIP/dataloaders/classes/caltech101.txt",
                k_shot=k_shot,
            )
            print(f" Train images: {len(train)}, Test images: {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
        
    elif dataset_name == "dtd":
        SPLIT = str(kwargs['dtd_split'])
        print(f" Using split: [{SPLIT}/10]")
        test = DTD(root="", 
            train=False,
            transform=transform,
            SPLIT=SPLIT,
        )
        
        if loader_type=="test": 
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type=="train":
            train = DTD(root="", 
                train=True,
                transform=transform_train,
                SPLIT=SPLIT,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
        
    elif dataset_name == "food101":
        test = Food101(root="", train=False, transform=transform)
        
        if loader_type == "test": 
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type=="train":
            train = Food101(root="", train=True, transform=transform_train, k_shot=k_shot)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "fgvc_aircraft":
        test = aircraft(root="", 
            train=False,
            transform=transform,
        )
        
        if loader_type == "test": 
            print(f" Test images {len(test)}")
            return test

        elif loader_type == "train": 
            train = aircraft(root="", 
                train=True,
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "sun397":
        test = SUN397(root="", train=False, transform=transform)
        
        if loader_type == "test": 
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = SUN697(path="./datasets/SUN397", train=True, transform=transform_train, k_shot=k_shot)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "pets":
        test = OxfordPets(root="./datasets/oxford_pets", train=False, transform=transform)
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = OxfordPets(path="./datasets/oxford_pets", train=True, transform=transform_train, k_shot=k_shot)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "cars":
        test = Cars(root="./datasets/stanford_cars", train=False, transform=transform)
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = Cars(path="./datasets/stanford_cars", train=True, transform=transform_train, k_shot=k_shot)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == "flowers":
        test = OxfordFlowers(root="./datasets/Flowers102/flowers-102", train=False, transform=transform)
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = OxfordFlowers(path="./datasets/Flowers102/flowers-102", 
                train=True, 
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "eurosat":
        test = EuroSAT(root="./datasets/EuroSAT", train=False, transform=transform)
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = EuroSAT(path="./datasets/EuroSAT", train=True, transform=transform_train, k_shot=k_shot)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == "ucf101":
        test = UCF101(root="./datasets/UCF101_midframes", 
            train=False, 
            transform=transform,
        )
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = UCF101(path="./datasets/UCF101_midframes", 
                train=True, 
                transform=transform_train,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    else:
        raise AttributeError(f"{dataset_name} is not currently supported.")