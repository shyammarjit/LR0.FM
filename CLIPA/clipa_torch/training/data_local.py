import ast
import json
import logging
import math
import os
import random
import sys
import time
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from .dataset import IterableImageDataset
from .device_env_factory import use_xla


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if hasattr(self.dataloader, 'dataset') and hasattr(self.dataloader.dataset, 'set_epoch'):
            self.dataloader.dataset.set_epoch(epoch)
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, weights=args.train_data_upsampling_factors, deterministic=True, epoch=shared_epoch)]
    else:
        assert args.train_data_upsampling_factors is None, "--train_data_upsampling_factors is only supported when sampling with replacement (together with --dataset-resampled)."
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class SyntheticDataset_LR(Dataset):
    def __init__(self, root, args, transform, tokenize):
        self.args = args
    
        self.load_2k = True 
        self.load_7k = True 

        self.load_7k_1K = False
        self.load_7k_3K = False     
        self.load_7k_5K = True    
        self.load_7k_7K = False
        self.load_7k_9K = False

        
        self.multi_scale = True 
        self.multi_scale_16 = False   
        self.multi_scale_16_64 = False 
        self.multi_scale_16_128 = False

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
        

def get_synthetic_dataset_LR(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset_LR(root=args.train_data, args=args, transform=preprocess_fn, tokenize=tokenizer)
    num_samples = len(dataset)

    assert is_train
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

    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



def get_iterable_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, root=None):
    if root is None:
        root = args.train_data if is_train else args.val_data
    assert root
    if 'laion' in root:
        split = 'full' # laion400m uses full
        name = None
        input_name = 'jpg'
        target_name = 'txt'
    elif 'imagenet' in root:
        split = 'train' if is_train else 'validation'
        name = 'imagenet2012:5.1.0'
        input_name = 'image'
        target_name = 'label'
    else:
        raise NotImplementedError
    dataset = IterableImageDataset(
        root,
        name=name,
        reader='tfds',
        split=split,
        is_training=is_train,
        batch_size=args.batch_size,
        seed=args.seed,
        input_name=input_name,
        target_name=target_name,
        transform=preprocess_fn,
        target_transform=None,
        epoch=epoch,
        tokenizer=tokenizer,
        single_replica=not is_train,
        train_num_samples=args.train_num_samples
    )
    # give Iterable datasets early knowledge of num_workers so that sample estimates
    # are correct before worker processes are launched
    dataset.set_loader_cfg(num_workers=args.workers if args.workers != 0 else 1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # tfds has a internal shuffling mechanism
        num_workers=args.workers,
        persistent_workers=True if args.workers > 0 else False,
    )

    dataloader.num_samples = len(dataset) * args.world_size
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "LR":
        return get_synthetic_dataset_LR
    elif dataset_type == 'tfrecord':
        return get_iterable_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        if args.imagenet_val.startswith("gs://"):
            data["imagenet-val"] = get_iterable_dataset(args, preprocess_val, is_train=False, root=args.imagenet_val)
        else:
            data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
