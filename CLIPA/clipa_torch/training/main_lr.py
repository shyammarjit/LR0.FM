import os
import sys 

previous_dir = os.getcwd() # Folder A
sys.path.append(previous_dir)
from zero_shot import create_dataset_zero_shot, zeroshot_classifier, accuracy, get_classes_prompts

import glob
import logging

import re
import subprocess
import sys
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from functools import partial

try:
    import torch_xla.core.xla_model as xm
    import torch_xla
    _HAS_XLA = True
except ImportError as e:
    xm = None
    torch_xla = None
    _HAS_XLA = False

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

from open_clip_local import create_model_and_transforms
from training.data_local import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.device_env_factory import use_xla, state_dict_to_cpu
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate, train_SS_MS
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync

import torch.nn.functional as F
from urllib.request import urlopen
from PIL import Image
from open_clip import get_tokenizer, create_loss



LATEST_CHECKPOINT_NAME = "epoch_latest.pt"




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


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None



def accuracy(output, target, topk=(1,)):
    """
    Zero-shot prediction. This is taken form CLIP official codebase.
    Please refer to .
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def eval_fn(loader, model, zeroshot_weights, logging=None, **kwargs):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(loader):
            images = images.cuda() # torch.Size([400, 3, 224, 224])
            target = target.cuda() # torch.Size([400])
            
            # predict
            image_features = model.encode_image(images, **kwargs) # torch.Size([400, 512]
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



def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            assert not use_xla()
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):

        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        os.system(f"rm -rf {args.log_path}")
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    print(args.tensorboard)
    
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if args.copy_codebase and is_master(args):
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
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

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)

    if args.model == "ViT-L-14-CLIPA":
        args.model = 'hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B'
    elif args.model == "ViT-H-14-CLIPA-datacomp1B":
        args.model = 'hf-hub:UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B'
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
        lr_mode=  args.lr_mode,
        strict=False,
    )

    random_seed(args.seed, args.rank)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    
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

    if args.distributed and not args.horovod and not use_xla():
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
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

        if use_xla():
            assert args.precision in ['fp32', 'bf16'], 'currently pytorch xla does not work with amp!'
        else:
            scaler = GradScaler() if "amp" in args.precision else None

    logging.info(f"**** Trainable: {gain_or_bias_params_names}")
    logging.info(f"**** Trainable: {rest_params_names}")

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    args.val_data = None
    args.imagenet_val = None
    args.imagenet_v2 = None
    args.dataset_type = "LR"
    
    tokenizer = get_tokenizer(args.model)
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=tokenizer,)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

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
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(model, data, start_epoch, -1, args, True, True, writer)
        return

    loss = create_loss(args)




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
    test = create_dataset_zero_shot(args.dataset, dtd_split=1, low_resolution=args.low_resolution,\
        org_resolution=336 if "336" in args.model else 224, root=args.dataset_dir, 
    )
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)

    # Creating zero-shot classifier weights
    zeroshot_weights = zeroshot_classifier(classes, templates, model.module, tokenizer) # torch.Size([512, 1000])

    if not args.debug:
        if args.lr_mode:
            eval_fn(loader, model.module, zeroshot_weights, use_spatial_tokens=False, logging=logging)
        else:
            eval_fn(loader, model.module, zeroshot_weights, logging=logging)
    BEST_1 = 0
    BEST_5 = 0
    save = True
    
    #################### 
    

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')


    
        logging.info(f"train_SS_MS")
        train_SS_MS(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        # train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        
        if args.lr_mode:
            top1, top5 = eval_fn(loader, model.module, zeroshot_weights, use_spatial_tokens=True, logging=logging)
        else:
            top1, top5 = eval_fn(loader, model.module, zeroshot_weights, logging=logging)
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

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

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


def _mp_entry(*args):
    main(sys.argv[1:])

if __name__ == "__main__":
    main(sys.argv[1:])
