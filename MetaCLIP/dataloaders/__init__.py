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

def get_dataloader(dataset_name, transform=None, loader_type = "test", root=None, **kwargs):
    print(f" Dataset: {dataset_name.upper()}.")
    print(f" Transformation: {transform}")
    print(f" Dataloader type: {loader_type}.")
    
    if dataset_name == "cifar10":
        root = os.path.expanduser("~/.cache")
        test = CIFAR10(root, download=True, train=False, transform=transform)
        if loader_type=="test":
            print(f" Test images: {len(test)}"); return test
        elif loader_type=="train":
            train = CIFAR10(root, download=True, train=True, transform=transform)
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
            train = CIFAR100(root, download=True, train=True, transform=transform)
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
            train = STL10(root, download=True, split='train', transform=transform)
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
            train = MNIST(root, download=True, train=True, transform=transform)
            print(f" Train images: {len(train)}, Test images: {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "imagenet1k":
        test = Imagenet(root="./datasets/ImageNet", 
            train=False,
            class_info="./MetaCLIP/dataloaders/imagenet.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Imagenet(root="./datasets/ImageNet", 
                train=True,
                class_info="./MetaCLIP/dataloaders/imagenet.txt",
                transform=transform,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == 'imagenet_a':
        test = ImagenetA(root="./datasets/imagenet-a", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_a.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = ImagenetA(root="./datasets/imagenet-a", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_a.txt",
                transform=transform,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == 'imagenet_r':
        test = ImagenetR(root="./datasets/imagenet-r", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_r.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = ImagenetR(root="./datasets/imagenet-r", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_r.txt",
                transform=transform,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == 'imagenet_sketch':
        test = Imagenet_sketch(root="./datasets/imagenet_sketch", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_sketch.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Imagenet_sketch(root="./datasets/imagenet_sketch", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_sketch.txt",
                transform=transform,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == 'imagenet_v2':
        test = Imagenet_V2(root="./datasets/imagenetv2/tree/main/imagenetv2-matched-frequency", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_v2.txt",
            transform=transform,
        )
        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Imagenet_V2(root="./datasets/imagenetv2/tree/main/imagenetv2-matched-frequency", 
                train=True,
                class_info="./CLIP/dataloaders/imagenet_v2.txt",
                transform=transform,
                k_shot=k_shot,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")    





    elif dataset_name == "caltech101":
        test = Caltech101(root="./datasets/caltech-101/", 
            train=False,
            transform=transform,
            categories="./MetaCLIP/dataloaders/classes/caltech101.txt",
        )

        if loader_type=="test":
            print(f" Test images: {len(test)}")
            return test
        elif loader_type=="train":
            train = Caltech101(root="./datasets/caltech-101/", 
                train=True,
                transform=transform,
                categories="./MetaCLIP/dataloaders/classes/caltech101.txt",
            )
            print(f" Train images: {len(train)}, Test images: {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
        
    elif dataset_name == "dtd":
        SPLIT = str(kwargs['dtd_split'])
        print(f" Using split: [{SPLIT}/10]")
        test = DTD(root="./datasets/dtd/", 
            train=False,
            transform=transform,
            SPLIT=SPLIT,
        )
        
        if loader_type=="test": 
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type=="train":
            train = DTD(root="./datasets/dtd/", 
                train=True,
                transform=transform,
                SPLIT=SPLIT,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
        
    elif dataset_name == "food101":
        root = root if root else "./datasets/food-101/"
        test = Food101(root=root, train=False, transform=transform)
        
        if loader_type == "test": 
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type=="train":
            train = Food101(root=root, train=True, transform=transform)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "fgvc_aircraft":
        test = aircraft(root="./datasets/fgvc-aircraft-2013b/data/", 
            train=False,
            transform=transform,
        )
        
        if loader_type == "test": 
            print(f" Test images {len(test)}")
            return test

        elif loader_type == "train": 
            train = aircraft(path="./datasets/fgvc-aircraft-2013b/data/", 
                train=True,
                transform=transform,
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "sun397":
        test = SUN397(root="./datasets/SUN397/", train=False, transform=transform)
        
        if loader_type == "test": 
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = SUN697(path="./datasets/SUN397", train=True, transform=transform)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "pets":
        if root is None :
            root="./datasets/oxford_pets"    
        test = OxfordPets(root=root, train=False, transform=transform)
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = OxfordPets(path="./datasets/oxford_pets", train=True, transform=transform)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    
    elif dataset_name == "cars":
        root = root if root else "./datasets/stanford_cars"
        test = Cars(root=root, train=False, transform=transform)
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = Cars(path="./datasets/stanford_cars", train=True, transform=transform)
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")

    elif dataset_name == "flowers":
        root = root if root else "./datasets/Flowers102/flowers-102"
        test = OxfordFlowers(root=root, train=False, transform=transform)
        
        if loader_type == "test":
            print(f" Test images {len(test)}")
            return test
        
        elif loader_type == "train": 
            train = OxfordFlowers(path="./datasets/Flowers102/flowers-102", train=True, transform=transform)
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
            train = EuroSAT(path="./datasets/EuroSAT", train=True, transform=transform)
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
                transform=transform
            )
            print(f" Train images {len(train)}, Test images {len(test)}")
            return train, test
        else:
            raise ValueError(f" Wrong {loader_type} type.")
    else:
        raise AttributeError(f"{dataset_name} is not currently supported.")