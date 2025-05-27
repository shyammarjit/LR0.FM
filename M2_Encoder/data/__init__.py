import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .imagenet_a import ImagenetA
from .imagenet_r import ImagenetR
from .imagenet_sketch import Imagenet_sketch
from .imagenet_v2 import Imagenet_V2

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def create_transformation_zero_shot(low_resolution=128, org_resolution=224):
    """
    Normalization should not be there
    """
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))      
    if low_resolution!=org_resolution:
        transform_test = transforms.Compose([
            transforms.Resize((low_resolution, low_resolution),interpolation=InterpolationMode.BICUBIC),
            transforms.Resize(size=(org_resolution, org_resolution), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor()
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(size=(org_resolution, org_resolution), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor()
        ])
        
    print("test transformations: ", transform_test)
    return transform_test


def create_dataset_zero_shot(dataset, min_scale=0.5, low_resolution=128, dtd_split=1, org_resolution=224):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))      
    if low_resolution!=org_resolution:
        transform_test = transforms.Compose([
            transforms.Resize((low_resolution, low_resolution),interpolation=InterpolationMode.BICUBIC),
            transforms.Resize(size=(org_resolution, org_resolution), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor()
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(size=(org_resolution, org_resolution), interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor()
        ])
        
    print("test transformations: ", transform_test)
    
    if dataset=='imagenet1k':
        from data.imagenet import Imagenet
        test = Imagenet(root="", 
            train=False,
            class_info="./Ant-Multi-Modal-Framework/prj/M2_Encoder/data/imagenet.txt",
            transform=transform_test,
        )

    elif dataset == 'imagenet_a':
        test = ImagenetA(root="", 
            train=False,
            class_info="./resolution-bm/CLIP/dataloaders/imagenet_a.txt",
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_r':
        test = ImagenetR(root="", 
            train=False,
            class_info="./resolution-bm/CLIP/dataloaders/imagenet_r.txt",
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_sketch':
        test = Imagenet_sketch(root="", 
            train=False,
            class_info="./resolution-bm/CLIP/dataloaders/imagenet_sketch.txt",
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_v2':
        test = Imagenet_V2(root="", 
            train=False,
            class_info="./resolution-bm/CLIP/dataloaders/imagenet_v2.txt",
            transform=transform_test,
        )
    
    elif dataset=='caltech101':
        from data.caltech101 import Caltech101
        test = Caltech101(root="", 
            train=False,
            transform=transform_test,
            categories="./resolution-bm/Ant-Multi-Modal-Framework/prj/M2_Encoder/data/classes/caltech101.txt",
        )

    elif dataset=='dtd':
        from data.dtd import DTD
        print(f" Using split: [{dtd_split}/10]")
        test = DTD(root="", train=False, transform=transform_test, SPLIT=str(dtd_split))
    
    elif dataset=='food101':
        from data.food101 import Food101
        test = Food101(root="", train=False, transform=transform_test)
    
    elif dataset=='sun397':
        from data.sun397 import SUN397
        test = SUN397(root="", train=False, transform=transform_test)
    
    elif dataset=='cars':
        from data.cars import Cars
        test = Cars(root="", train=False, transform=transform_test)
    
    elif dataset=='fgvc_aircraft':
        from data.aircraft import aircraft
        test = aircraft(root="", train=False, transform=transform_test)
    
    elif dataset == "pets":
        from data.oxford_pets import OxfordPets
        test = OxfordPets(root="", train=False, transform=transform_test)
    
    elif dataset == "flowers":
        from data.flowers import OxfordFlowers
        test = OxfordFlowers(root="", train=False, transform=transform_test)
    
    elif dataset == "eurosat":
        from data.eurosat import EuroSAT
        test = EuroSAT(root="", train=False, transform=transform_test)
    
    elif dataset == "ucf101":
        from data.ucf101 import UCF101
        test = UCF101(root="", train=False, transform=transform_test)
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