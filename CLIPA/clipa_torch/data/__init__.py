import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .imagenet_a import ImagenetA
from .imagenet_r import ImagenetR
from .imagenet_sketch import Imagenet_sketch
from .imagenet_v2 import Imagenet_V2
import os 

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def create_transformation_zero_shot(low_resolution=128, org_resolution=336):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))      
    if low_resolution!=org_resolution:
        transform_test = transforms.Compose([
            transforms.Resize(low_resolution,interpolation=InterpolationMode.BICUBIC),
            # transforms.Resize((low_resolution, low_resolution),interpolation=InterpolationMode.BICUBIC),
            transforms.Resize(org_resolution,interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(org_resolution, org_resolution)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(org_resolution,interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(org_resolution, org_resolution)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    return transform_test
    
def create_dataset_zero_shot(dataset, min_scale=0.5, low_resolution=128, dtd_split=1, org_resolution=336, root=None, enable_resize=True):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))      
    if low_resolution!=org_resolution:
        if enable_resize:
            transform_test = transforms.Compose([
                transforms.Resize(low_resolution,interpolation=InterpolationMode.BICUBIC),
                transforms.Resize(org_resolution,interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=(org_resolution, org_resolution)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(low_resolution,interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=(low_resolution, low_resolution)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalize,
            ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(org_resolution,interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(org_resolution, org_resolution)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    
    print("test transformations: ", transform_test)
    
    if dataset=='imagenet1k':
        class_info="~/resolution-bm/EVA/EVA-CLIP/rei/data/imagenet.txt"
        home_directory = os.path.expanduser('~')
        class_info = class_info.replace("~", home_directory)
        
        from data.imagenet import Imagenet
        test = Imagenet(root="./datasets/ImageNet", 
            train=False,
            class_info=class_info,
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_a':
        class_info="~/resolution-bm/CLIP/dataloaders/imagenet_a.txt"
        home_directory = os.path.expanduser('~')
        class_info = class_info.replace("~", home_directory)
        test = ImagenetA(root="./datasets/imagenet-a", 
            train=False,
            class_info=class_info,
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_r':
        class_info="~/resolution-bm/CLIP/dataloaders/imagenet_r.txt"
        home_directory = os.path.expanduser('~')
        class_info = class_info.replace("~", home_directory)
        test = ImagenetR(root="./datasets/imagenet-r", 
            train=False,
            class_info=class_info,
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_sketch':
        class_info="~/resolution-bm/CLIP/dataloaders/imagenet_sketch.txt"
        home_directory = os.path.expanduser('~')
        class_info = class_info.replace("~", home_directory)
        
        test = Imagenet_sketch(root="./datasets/imagenet_sketch", 
            train=False,
            class_info=class_info,
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_v2':
        class_info="~/resolution-bm/CLIP/dataloaders/imagenet_v2.txt"
        home_directory = os.path.expanduser('~')
        class_info = class_info.replace("~", home_directory)
        
        test = Imagenet_V2(root="./datasets/imagenetv2/tree/main/imagenetv2-matched-frequency", 
            train=False,
            class_info=class_info,
            transform=transform_test,
        )
    
    elif dataset=='caltech101':
        categories="./EVA/EVA-CLIP/rei/data/classes/caltech101.txt"
        home_directory = os.path.expanduser('~')
        categories = categories.replace("~", home_directory)
        from data.caltech101 import Caltech101
        test = Caltech101(root="./datasets/caltech-101/", 
            train=False,
            transform=transform_test,
            categories=categories,
        )

    elif dataset=='dtd':
        from data.dtd import DTD
        print(f" Using split: [{dtd_split}/10]")
        test = DTD(root="./datasets/dtd/", train=False, transform=transform_test, SPLIT=str(dtd_split))
    
    elif dataset=='food101':
        from data.food101 import Food101
        root = root if root else "./datasets/food-101/"
        test = Food101(root=root, train=False, transform=transform_test)
    
    elif dataset=='sun397':
        from data.sun397 import SUN397
        test = SUN397(root="./datasets/SUN397/", train=False, transform=transform_test)
    
    elif dataset=='cars':
        from data.cars import Cars
        test = Cars(root="./datasets/stanford_cars", train=False, transform=transform_test)
    
    elif dataset=='fgvc_aircraft':
        from data.aircraft import aircraft
        root = root if root else "./datasets/fgvc-aircraft-2013b/data/"
        test = aircraft(root=root, train=False, transform=transform_test)
    
    elif dataset == "pets":
        from data.oxford_pets import OxfordPets
        root = root if root else "./datasets/oxford_pets"
        test = OxfordPets(root=root, train=False, transform=transform_test)
    
    elif dataset == "flowers":
        from data.flowers import OxfordFlowers
        root = root if root else "./datasets/Flowers102/flowers-102"
        test = OxfordFlowers(root=root, train=False, transform=transform_test)
    
    elif dataset == "eurosat":
        from data.eurosat import EuroSAT
        test = EuroSAT(root="./datasets/EuroSAT", train=False, transform=transform_test)
    
    elif dataset == "ucf101":
        from data.ucf101 import UCF101
        test = UCF101(root="./datasets/UCF101_midframes", train=False, transform=transform_test)
    else:
        raise ValueError(f"{dataset} not supported.")
    
    print(f"No of images: {len(test)}")
    return test 

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders