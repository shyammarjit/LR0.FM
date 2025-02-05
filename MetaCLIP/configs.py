import os
import inspect

from collections import OrderedDict
from dataclasses import dataclass

import sys
sys.path.append("src")

from training.params import get_default_params


@dataclass
class Config:
    train_data = None
    val_data = None
    train_num_samples = None
    val_num_samples = None
    dataset_type = "auto"
    dataset_resampled = False
    csv_separator = "\t"
    csv_img_key = "filepath"
    csv_caption_key = "title"
    imagenet_val = "/datasets01/imagenet_full_size/061417/val"
    imagenet_v2 = None
    logs = "./logs/"
    log_local = False
    name = None
    workers = 8
    batch_size = 64
    epochs = 32
    lr = None
    beta1 = None
    beta2 = None
    eps = None
    wd = 0.2
    warmup = 2000  # 10000
    use_bn_sync = False
    skip_scheduler = False
    save_frequency = 1
    save_most_recent = True  # False
    zeroshot_frequency = 1
    val_frequency = 1
    resume = None
    precision = "amp"
    clip_model = "CLIP"
    model = "RN50"
    pretrained = ''
    pretrained_image = False
    lock_image = False
    lock_image_unlocked_groups = 0
    lock_image_freeze_bn_stats = False
    grad_checkpointing = False
    local_loss = False
    gather_with_grad = False
    force_quick_gelu = False
    torchscript = False
    trace = False
    dist_url = "env://"
    dist_backend = "nccl"
    report_to = ""
    wandb_notes = ''
    debug = False
    copy_codebase = False
    horovod = False
    ddp_static_graph = False
    no_set_device_rank = False
    seed = 0
    norm_gradient_clip = None

    def __post_init__(self):
        args = self
        args.name = self.__class__.__name__
        args.output_dir = os.path.join(args.logs, args.name)

        for name, val in get_default_params(args.model).items():
            if getattr(args, name) is None:
                setattr(args, name, val)


def parse_start_end(shards):
    start, end = os.path.basename(shards).split("{")[1].split("}")[0].split("..")
    return int(start), int(end)


def search_config(config_name):
    import importlib
    project_dir = os.path.dirname(__file__)
    all_configs = {}
    for code in os.listdir(project_dir):
        if code.endswith(".py") and code.startswith("run_configs"):
            module = importlib.import_module(code[:-3])
            for _config_name in dir(module):
                if _config_name in ["Config"] or _config_name.startswith("__") or _config_name.startswith("run_config"):
                    continue
                if _config_name not in all_configs:
                    all_configs[_config_name] = module
    print(f"launching {config_name} from {all_configs[config_name].__file__}")
    config = getattr(all_configs[config_name], config_name)()
    return config
