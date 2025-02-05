# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import mmap
import os
import json

from typing import Any, Callable, Optional

import numpy as np
import random

import tarfile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging 

from io import BytesIO
from PIL import Image, ImageDraw, ImageFilter
from training.distributed import world_info_from_env
from torchvision.utils import save_image
from open_clip_local import tokenize
from .data_local import DataInfo



def normalize(x):
    return (x - x.min()) / (x.max() - x.min())



class IterativeWebDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, transform, tokenize):
        self.args = args
        start, end = os.path.basename(args.train_data).split("{")[1].split("}")[0].split("..")
        self.num_shards = int(end) - int(start)
        self.root_dir = os.path.dirname(args.train_data)
        self.transform = transform
        self.tokenizer = tokenize
        self.start_shard_id = 0
        self.shard_ids = list(range(self.num_shards))

    def set_epoch(self, epoch, num_batches, step=0):
        random.seed(epoch+step)
        self.shard_ids = list(range(self.num_shards))
        random.shuffle(self.shard_ids)
        self.start_shard_id = (num_batches * epoch) % self.num_shards

    def _get_tarball_path(self, shard_id):
        return os.path.join(self.root_dir, f"{shard_id % 100}", f"{shard_id}.tar")

    def _get_next_shard_id(self, shard_id):
        shard_id += self.group_size
        return shard_id % self.num_shards

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        _, global_rank, world_size = world_info_from_env()
        self.group_size = int(num_workers * world_size)
        shard_id = num_workers * global_rank + worker_id
        shard_id = (shard_id + self.start_shard_id) % self.num_shards
        shard_id = self.shard_ids[shard_id]

        while True:
            tarball_path = self._get_tarball_path(shard_id)
            if not os.path.exists(tarball_path):
                shard_id = self._get_next_shard_id(shard_id)
                continue

            with tarfile.open(tarball_path) as tar:
                members = tar.getmembers()

                # metaclip_v1 can be iterative but the paper uses mmap for random access.
                json_uuid, img_uuid = -1, -2
                for member in members:
                    if member.name.endswith(".json"):
                        json_uuid = member.name[:-len(".json")]
                        with tar.extractfile(member) as f:
                            text_json = json.load(f)

                    if member.name.endswith(".jpeg"):
                        img_uuid = member.name[:-len(".jpeg")]
                        with tar.extractfile(member) as f:
                            img = f.read()

                    if img_uuid != json_uuid:
                        # assume uuid is json even and img ord;
                        continue

                    txt = random.choice(text_json["texts"])[1]
                    txt = self.tokenizer([txt])[0]

                    with Image.open(BytesIO(img)) as img:
                        image = img.convert("RGB")
                        image = self.transform(image)

                    yield image, txt

            shard_id = self._get_next_shard_id(shard_id)


def get_metaclip_iter_wds_dataset(args, preprocess_fn, is_train, epoch=0):
    # borrowed from get_csv_dataset
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = IterativeWebDataset(
        args,
        preprocess_fn,
        tokenize,
    )

    assert is_train
    num_samples = args.train_num_samples
    sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
    )

    dataloader.num_samples = num_samples
    dataloader.num_batches = int(num_samples / (args.batch_size * args.world_size))

    return DataInfo(dataloader, sampler)






class IterativeWebDataset_LR(Dataset):
    def __init__(self, root, args, transform, tokenize):
        self.args = args
        

        self.load_2k = True 
        self.load_7k = True 

        self.load_7k_1K = False
        self.load_7k_3K = True     
        self.load_7k_5K = False    
        self.load_7k_7K = False
        self.load_7k_9K = False

        
        self.multi_scale = True 
        self.multi_scale_16 = False   
        self.multi_scale_16_64 = False 
        self.multi_scale_16_128 = True    

        self.transform = transform
        
        self.images = [] 
        self.labels = []
        index = []

        if self.load_7k:
            with open (f"{root}/caption_2k.txt", "r") as f:
                Lines = f.readlines()[:-1]
            with open (f"{root}/caption_5k.txt", "r") as f:
                Lines += f.readlines()
        elif self.load_2k:
            with open (f"{root}/caption_2k.txt", "r") as f:
                Lines = f.readlines()[:-1]
            
         
        captions = [e.strip() for e  in Lines] 
        # self.string_captions  = [e for e in captions] 
        # self.hr_captions  = [preprocess_txt(e) for e in captions] 
        # self.lr_captions  = [preprocess_txt("low resolution image of " + e) for e in captions] 
        # self.lr_captions1  = [preprocess_txt("Pixelated image of " + e) for e in captions] 
        # self.lr_captions2  = [preprocess_txt("Downsampled image of " + e) for e in captions] 
        # self.lr_captions3  = [preprocess_txt("Low Quality image of  " + e) for e in captions] 

        if self.load_7k:
            samples_2k = os.path.join(root, "Mul_samples_50_2K")
            samples_5k = os.path.join(root, "Mul_samples_50_5K")
            data_srcs = [samples_2k, samples_5k]
            flags_5K = [0, 1]
        elif self.load_2k:
            samples_10 = os.path.join(root, "Mul_samples_10_2K")
            samples_30 = os.path.join(root, "Mul_samples_30_2K")
            samples_50 = os.path.join(root, "Mul_samples_50_2K")
            data_srcs = [samples_10, samples_30, samples_50]
            
        if self.load_7k:
            for k,series in enumerate(data_srcs):
                flag_5k = flags_5K[k]
                for folder in os.listdir(series):
                    if self.load_7k_1K and int(folder) > 10:
                        continue
                    elif self.load_7k_3K and int(folder) > 30:
                        continue                 
                    folder_path = os.path.join(series, folder)
                    max_range = 2000
                    start_index = 0
                    if flag_5k:
                        max_range = 5000
                        start_index += 2000
                    for i in range(max_range):
                        img = f"rohit_caption_{i}.png"
                        img_path = os.path.join(folder_path, img)
                        assert os.path.exists(img_path)
                        self.images.append( img_path )
                        self.labels.append( start_index + int(img.split("_")[-1][:-4]) )
        elif self.load_2k:
            done = 0
            
            if self.load_7k_1K:to_be_completed = 10 
            elif self.load_7k_3K:to_be_completed = 30
            elif self.load_7k_5K:to_be_completed = 50
            elif self.load_7k_7K:to_be_completed = 70
            elif self.load_7k_9K:to_be_completed = 90

            for k,series in enumerate(data_srcs):
                if done == to_be_completed: break 
                for folder in os.listdir(series):
                    if done == to_be_completed: break 
                    done += 1
                    folder_path = os.path.join(series, folder)
                    max_range = 2000
                    start_index = 0
                    for i in range(max_range):
                        img = f"rohit_caption_{i}.png"
                        img_path = os.path.join(folder_path, img)
                        assert os.path.exists(img_path)
                        self.images.append( img_path )
                        self.labels.append( start_index + int(img.split("_")[-1][:-4]) )

        self.train_num_samples = len(self.images)
        
        if self.multi_scale or self.multi_scale_random:
            self._resize_shapes = [range(16, 32), range(32, 64), range(64, 128), range(128, 224)] 
        
        if self.multi_scale_16:
            self._resize_shapes = [range(16, 32)] 
        elif self.multi_scale_16_64:
            self._resize_shapes = [range(16, 32), range(32, 64)] 
        elif self.multi_scale_16_128:
            self._resize_shapes = [range(16, 32), range(32, 64), range(64, 128)] 

        print( f"**** {self._resize_shapes} ****")
        print( f"**** Captions : {len(self.labels)} Images : {len(self.images)} ****")

        logging.info(f"**** {self._resize_shapes} ****")
        logging.info( f"**** Captions : {len(self.labels)} Images : {len(self.images)} ****")

        idx = 0
        img, label = self.__getitem__(idx)
        # save_image(normalize(img), "img.png")

    def __len__(self):
        return self.train_num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        img_hr = Image.open(img).convert('RGB')

        H,W = img_hr.size
        
        if self.multi_scale:
            lr_shapes = [random.choice(e) for e in self._resize_shapes]
            img_lrs = [img_hr.resize((lr_shape, lr_shape)).resize((H, W)) for lr_shape in lr_shapes]
            img_hr = self.transform(img_hr)
            img_lr = [self.transform(img_lr) for img_lr in img_lrs]
            img = torch.stack([img_hr] + img_lr)
        elif self.multi_scale_random:
            lr_shapes = random.choice( self._resize_shapes )
            lr_shapes = random.choice( lr_shapes )
            img_lr = img_hr.resize((lr_shapes, lr_shapes)).resize((H, W))

            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)

            img = torch.stack([img_hr, img_lr])
        else:
            lr_shape = random.choice(self._resize_shapes)
            img_lr = img_hr.resize((lr_shape, lr_shape)).resize((H, W))
            # img_hr.save("temp.png") , img_lr.save("temp2.png")
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
        
            img = torch.stack([img_hr, img_lr])

        return img, label
        

    

def get_metaclip_LR(args, preprocess_fn, is_train, epoch=0):
    # borrowed from get_csv_dataset
    root = args.train_data if is_train else args.val_data
    dataset = IterativeWebDataset_LR(
        root, 
        args,
        preprocess_fn,
        tokenize,
    )

    assert is_train
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None
    
    if args.train_num_samples != 2 and args.train_num_samples is not None:
        from .sampler import DistributedRandomIdentitySampler
        sampler = DistributedRandomIdentitySampler(dataset.labels, num_instances=args.train_num_samples, seed=1)
        
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=False,
        sampler=sampler,
        drop_last=is_train,
    )
    
    print(f"*****  {len(dataloader)} *****")
    # *****  1531 *****
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



