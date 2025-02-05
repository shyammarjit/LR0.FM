# Copyright (c) Meta Platforms, Inc. and affiliates

import sys
sys.path.append("src")

import os
previous_dir = os.getcwd() # Folder A
sys.path.append(previous_dir)
from zero_shot import get_classes_prompts, get_dataloader, zeroshot_classifier






import re
import logging

import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler


try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip_local import create_model_and_transforms, trace_model, get_mean_std
from open_clip_local.model import CLIP, VisualTransformer, Transformer, ResidualAttentionBlock
from training.data_local import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate
from training import train






def save_checkpoint(model, optimizer, scaler, completed_epoch, args):
    checkpoint_dict = {
        "epoch": completed_epoch,
        "name": args.name,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()
    torch.save(
        checkpoint_dict,
        os.path.join(args.checkpoint_path, f"epoch_Best.pt"),
    )

    

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def accuracy(output, target, topk=(1,)):
    """
    Zero-shot prediction. This is taken form CLIP official codebase.
    Please refer to .
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def eval_fn(loader, model, zeroshot_weights, use_spatial_tokens=True , logging=None):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(loader):
            images = images.cuda() # torch.Size([400, 3, 224, 224])
            target = target.cuda() # torch.Size([400])
            
            # predict
            image_features = model.encode_image(images, use_spatial_tokens=use_spatial_tokens) # torch.Size([400, 512]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    logging.info(f"Top-1 accuracy: {top1:.2f}")
    logging.info(f"Top-5 accuracy: {top5:.2f}")
    return top1, top5

def main(args=None):
    open_clip = None

    if args is None:
        args_ = parse_args()

        from configs import search_config
        from copy import deepcopy

        config = search_config(args_.config_name)
        args = config 
        args.zeroshot_frequency = args_.zeroshot_frequency
        args.train_data = args_.train_data
        args.epochs = args_.epochs
        args.workers = args_.workers
        args.lr_mode = args_.lr_mode
        args.name = args_.name
        args.pretrained = args_.pretrained

        args.dist_backend = args_.dist_backend
        args.dist_url = args_.dist_url

        args.class_dir = args_.class_dir
        args.templates_dir = args_.templates_dir
        args.dataset_type = "LR"

        args.distributed = True 
        args.distributed_engine = args_.dist_backend
        # print(args.distributed)
        args.val_data = None
        args.imagenet_val = None 
        args.imagenet_v2 = None

        args.low_resolution = args_.low_resolution
        args.engine = args_.engine
        args.debug = args_.debug
        
        args.batch_size = args_.batch_size
        args.train_num_samples = args_.train_num_samples

        open_clip = args_.open_clip

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace('/', '-')

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # discover initial world args early so we can log properly
    # args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    
    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        os.system(f"rm -rf {args.log_path}")
        if os.path.exists(args.log_path) and args.resume is None and not hasattr(args, "eval"):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Set logger
    # args.log_level = logging.DEBUG if args.debug else logging.INFO
    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ['amp', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    mean, std = get_mean_std(args)

    if open_clip:
        # args.model = args.model.replace("-quickgelu", "")
        # args.force_quick_gelu = False 
        args.pretrained = '~/.cache/huggingface/hub/models--laion--CLIP-ViT-B-16-DataComp.XL-s13B-b90K/snapshots/d110532e8d4ff91c574ee60a342323f28468b287/open_clip_pytorch_model.bin'
        home_directory = os.path.expanduser('~')
        args.pretrained = args.pretrained.replace("~", home_directory)
        print(args.model)

    
    # print(args.__dict__)
    logging.info(f"***** Model : {args.model} \t Name: {args.name} \t Local Rank: {args.local_rank} \t  Rank: {args.rank} \t World Size: {args.world_size}")
    logging.info(f"***** Model : {args.model} \t Pretraind: {args.pretrained} \t")
    logging.info(f"***** precision={args.precision} \t device={args.device} \t jit={args.torchscript}, \t force_quick_gelu={args.force_quick_gelu} \t pretrained_image={args.pretrained_image}")
    inmem = hasattr(args, "inmem")
    logging.info(f"***** inmem={inmem} \t clip_model={args.clip_model}")

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        mean=mean, std=std,
        inmem=hasattr(args, "inmem"),
        clip_model=args.clip_model,
        lr_mode=  args.lr_mode,
        strict=False,
        image_resolution=args_.low_resolution,
    )

    random_seed(args.seed, args.rank)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # if args.distributed_engine == 'ddp':
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        # ddp_args['find_unused_parameters'] = True if "Alt" in args.clip_model or "Dot" in args.clip_model else False   # huxu
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        # else:
        #     print("--distrubted_engine should be either 'ddp'")
        #     sys.exit(1)

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        gain_or_bias_params_names = [n for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        rest_params_names = [n for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if args.precision == "amp":
            scaler = GradScaler()
        else:
            scaler = None

    logging.info(f"**** Trainable: {gain_or_bias_params_names}")
    logging.info(f"**** Trainable: {rest_params_names}")

    # optionally resume from a checkpoint
    start_epoch = 0
    start_epoch_step = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if next(iter(sd.items()))[0].startswith('_orig_mod'):
                    sd = {k[len('_orig_mod.'):]: v for k, v in sd.items()}
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                if 'epoch_step' in checkpoint:  # resuming a train checkpoint w/ epoch and optimizer state
                    start_epoch_step = checkpoint["epoch_step"] + 1
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch}, step {start_epoch_step})")
                else:
                    start_epoch_step = 0
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))


    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)
    assert len(data), 'At least one train or eval dataset must be specified.'

    if hasattr(args, "torchcompile") and args.torchcompile:
        logging.info('Compiling model...')
        try:
            model = torch.compile(model)
        except Exception:
            logging.warn("please use PyTorch 2.0")

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project="open-clip",
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        # if args.debug:
        #     wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data or hasattr(args, "eval") and args.eval:  # huxu: merge native/SLIP eval.
        # TODO: move to below first.
        from training.slip_evaluate import slip_evaluate
        from open_clip import tokenize
        # in case a downloaded model.
        os.makedirs(args.output_dir, exist_ok=True)
        slip_evaluate(args, model, preprocess_val, tokenize)
        evaluate(model, data, start_epoch, args, writer)
        return

    epoch_step = start_epoch_step








    #################### VALIDATION 
    args.dataset_dir = '/data/priyank/synthetic/oxford_pets'
    args.dataset = 'pets'
    if not os.path.exists(args.dataset_dir):
        args.dataset_dir = None
        args.dataset = 'imagenet_sketch'
    
    print("=====", args.class_dir, args.templates_dir)
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    # Prepare the dataloader
    test = get_dataloader(args.dataset, preprocess_val, loader_type="test", dtd_split=1, root=args.dataset_dir)
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)

    # Creating zero-shot classifier weights
    zeroshot_weights = zeroshot_classifier(classes, templates, model.module) # torch.Size([512, 1000])

    if not args.debug:
        eval_fn(loader, model.module, zeroshot_weights, use_spatial_tokens=False, logging=logging)
    # eval_fn(loader, model.module, zeroshot_weights, use_spatial_tokens=True)
    BEST_1 = 0
    BEST_5 = 0
    save = True
    
    #################### 
    
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        if hasattr(args, "engine"):
            engine = args.engine
            module = train
            logging.info(f"{engine}")
            engine_cls = getattr(module, engine)
            engine_cls(model, data, epoch, epoch_step, optimizer, scaler, scheduler, args, writer)
        else:
            train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)

        epoch_step = 0  # reset for next epoch.

        completed_epoch = epoch + 1
        if epoch % 2 ==0 :
            top1, top5 = eval_fn(loader, model.module, zeroshot_weights, use_spatial_tokens=True, logging=logging)
            if top5 > BEST_5:
                BEST_5 = top5
            if top1 >= BEST_1:
                save = True 
            if top1 > BEST_1:
                BEST_1 = top1
                save = True 

            if save:        
                save_checkpoint(model, optimizer, scaler, completed_epoch, args)
                save = False

            logging.info(f"Best Top-1 accuracy: {BEST_1:.2f} \t Best Top-5 accuracy: {BEST_5:.2f}")


    if hasattr(args, "eval") and args.eval and any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
        from training.slip_evaluate import slip_evaluate
        from open_clip import tokenize

        slip_evaluate(args, model, preprocess_val, tokenize)

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    import sys
    sys.path.append("./")
    main()
