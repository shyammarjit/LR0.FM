import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .imagenet_a import ImagenetA
from .imagenet_r import ImagenetR
from .imagenet_sketch import Imagenet_sketch
from .imagenet_v2 import Imagenet_V2

# from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
# from data.nocaps_dataset import nocaps_eval
# from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
# from data.vqa_dataset import vqa_dataset
# from data.nlvr_dataset import nlvr_dataset
# from data.pretrain_dataset import pretrain_dataset
# from transform.randaugment import RandomAugment

def _convert_image_to_rgb(image):
    return image.convert("RGB")

# def create_dataset(dataset, config, min_scale=0.5, low_resolution=384, original_resolution=384):
    
#     normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

#     transform_train = transforms.Compose([                        
#             transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
#             transforms.RandomHorizontalFlip(),
#             RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
#                                               'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
#             transforms.ToTensor(),
#             normalize,
#         ])        

#     if low_resolution != original_resolution:  
#         transform_test = transforms.Compose([
#             transforms.Resize((low_resolution, low_resolution),interpolation=InterpolationMode.BICUBIC),
#             transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     else:
#         transform_test = transforms.Compose([
#             transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
#             transforms.ToTensor(),
#             normalize,
#         ])
        
#     print("test transformations: ", transform_test)

#     if dataset=='pretrain':
#         dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)              
#         return dataset  
    
#     elif dataset=='caption_coco':   
#         train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'], prompt=config['prompt'])
#         val_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
#         test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
#         return train_dataset, val_dataset, test_dataset
    
#     elif dataset=='nocaps':   
#         val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
#         test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
#         return val_dataset, test_dataset   
    
#     elif dataset=='retrieval_coco':
#         # we are getting call here
#         train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
#         val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
#         test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
#         return train_dataset, val_dataset, test_dataset
    
#     elif dataset=='retrieval_flickr':
#         # we are getting call here
#         train_dataset = flickr30k_train(transform_train, config['image_root'], config['ann_root'])
#         val_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
#         test_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
#         return train_dataset, val_dataset, test_dataset     
    
#     elif dataset=='vqa': 
#         train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'], 
#                                     train_files = config['train_files'], split='train') 
#         test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
#         return train_dataset, test_dataset
    
#     elif dataset=='nlvr': 
#         train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'],'train')
#         val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'val')
#         test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'test')     
#         return train_dataset, val_dataset, test_dataset


def create_dataset_zero_shot(dataset, min_scale=0.5, low_resolution=128, original_resolution=384, dtd_split=1, root=None):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))      
    if low_resolution!=original_resolution:  
        transform_test = transforms.Compose([
            transforms.Resize((low_resolution, low_resolution),interpolation=InterpolationMode.BICUBIC),
            transforms.Resize((original_resolution,original_resolution),interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize((original_resolution,original_resolution),interpolation=InterpolationMode.BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
        
    print("test transformations: ", transform_test)
    
    if dataset=='imagenet1k':
        from data.imagenet import Imagenet
        test = Imagenet(root="", 
            train=False,
            class_info="./ALBEF/data/imagenet.txt",
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_a':
        test = ImagenetA(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_a.txt",
            transform=transform_test,
        )
        # may have readme file
    
    elif dataset == 'imagenet_r':
        test = ImagenetR(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_r.txt",
            transform=transform_test,
        )
        # may have readme file
    
    elif dataset == 'imagenet_sketch':
        test = Imagenet_sketch(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_sketch.txt",
            transform=transform_test,
        )
    
    elif dataset == 'imagenet_v2':
        test = Imagenet_V2(root="", 
            train=False,
            class_info="./CLIP/dataloaders/imagenet_v2.txt",
            transform=transform_test,
        )

    elif dataset=='caltech101':
        from data.caltech101 import Caltech101
        test = Caltech101(root="", 
            train=False,
            transform=transform_test,
            categories="./data/classes/caltech101.txt",
        )

    elif dataset=='dtd':
        from data.dtd import DTD
        print(f" Using split: [{dtd_split}/10]")
        test = DTD(root="", train=False, transform=transform_test, SPLIT=str(dtd_split))
    
    elif dataset=='food101':
        from data.food101 import Food101
        if root is None :
            root=""
        test = Food101(root=root, train=False, transform=transform_test)
    
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