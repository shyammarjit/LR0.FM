import logging
import os
import sys
import random
from datetime import datetime


previous_dir = os.getcwd() # Folder A
sys.path.append(previous_dir)
from data import create_dataset_zero_shot



import numpy as np
import torch
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from eva_clip import create_model_and_transforms, create_model_from_pretrained, trace_model, get_tokenizer

from training.data_train import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env, create_deepspeed_config
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import warmup_cosine_lr
from training.train import *
from training.optim import create_optimizer, get_all_parameters

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def read_txt(file_location):
    with open(file_location, 'r') as file:
        content = file.read(); content = str(content); content = content.split('\n', -1)
    try: content.remove("")
    except: pass
    return content

def get_classes_prompts(args):
    classes = read_txt(os.path.join(args.class_dir, f"{args.dataset}.txt"))
    templates = read_txt(os.path.join(args.templates_dir, f"{args.dataset}.txt"))
    return classes, templates

def zeroshot_classifier(classnames, templates, model, model_name=None):
    """ 
    Creating zero-shot classifier weights. This is taken form CLIP official codebase.
    Please refer to .
    """
    if model_name is not None: tokenizer = get_tokenizer(model_name)
    else: raise ValueError(f'wrong {model_name}')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenizer(texts).cuda() #tokenize
            class_embeddings = model.module.encode_text(texts)#embed with text encoder
            # print(class_embeddings.shape) # torch.Size([80, 512])
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            # print(class_embedding.shape) # torch.Size([512])
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        # print(zeroshot_weights.shape) # torch.Size([512, 1000])
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Zero-shot prediction. This is taken form CLIP official codebase.
    Please refer to .
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def evaluate(loader, model, zeroshot_weights, logging, use_spatial_tokens=True ):
    model = model.eval()
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(loader):
            images = images.cuda() # torch.Size([400, 3, 224, 224])
            target = target.cuda() # torch.Size([400])
            # from torchvision.utils import save_image
            # def normalize(x):return (x - x.min()) / (x.max() - x.min())
            # for i in range(len(images)):save_image(normalize(images[i]), f"Samples_{args.low_resolution}/{i}.png")
            
            # predict
            image_features = model.module.encode_image(images, use_spatial_tokens=use_spatial_tokens) # torch.Size([400, 512]
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
    args, ds_init = parse_args(args)

    if ds_init is not None:
        create_deepspeed_config(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True # cudnn error 
        
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
    else:
        args.name = '-'.join([
            args.name
        ])

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(args.log_path):
        #     print(
        #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
        #     )
        #     return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
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

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)
    model_mode=None 
    if args.e2e_fine_tuning:
        model_mode = -1
        if args.w_lr_tokens:
            model_mode = 2
    if args.robust_sam:
        model_mode = 3
    if args.lr_tokens_layer != -1:
        model_mode = args.lr_tokens_layer
    if args.vpt:
        model_mode = -2

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_clip=True,
        force_patch_dropout=args.force_patch_dropout,
        pretrained_image=args.pretrained_image,
        pretrained_text=args.pretrained_text,
        pretrained_visual_model=args.pretrained_visual_model,
        pretrained_text_model=args.pretrained_text_model,
        image_mean=args.image_mean,
        image_std=args.image_std,
        cache_dir=args.cache_dir,
        skip_list=args.skip_list,
        lr_clip=args.lr_mode,
        train_fn= args.train_fn,
        model_mode = model_mode,
        no_xattn=args.disable_xattn, 
    )
    # model, _, _ = create_model_and_transforms(args.backbone, pretrained, force_custom_clip=True)
    random_seed(args.seed, args.rank)

    total_n_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f'number of total params: {total_n_parameters}')

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'number of params with requires_grad: {n_parameters}')

    if hasattr(model, 'visual'):
        total_visual_n_parameters = sum(p.numel() for p in model.visual.parameters())
        logging.info(f'number of visual params: {total_visual_n_parameters}')
    if hasattr(model, 'text'):
        total_text_n_parameters = sum(p.numel() for p in model.text.parameters())
        logging.info(f'number of text params: {total_text_n_parameters}')

    model.to(device)
    model_without_ddp = model

    logging.info("Lock text tower...")
    model.lock_text_tower(
        unlocked_layers=args.lock_text_unlocked_layers,
        freeze_layer_norm=args.lock_text_freeze_layer_norm)

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

    # if args.distributed and not args.horovod:
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.enable_deepspeed:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args['static_graph'] = True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
            model_without_ddp = model.module
            

    # create optimizer and scaler
    optimizer = None
    scaler = None
    
    # model_mode
    if args.train_data or args.train_data_list or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'
                
        if not args.enable_deepspeed:
            scaler = GradScaler() if args.precision == "amp" else None
            optimizer = create_optimizer(args, model_without_ddp)
        else:
            scaler = None

            if args.optimizer != "lamb" and args.optimizer != "adamw":
                optimizer, optimizer_params = create_optimizer(
                    args,
                    model_without_ddp,
                    return_params=True)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
            else:
                optimizer_params = get_all_parameters(args, model)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
        if is_master(args, local=args.log_local):
            logging.info(f"num of optimizer.param_groups: {len(optimizer.param_groups)}")

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if args.enable_deepspeed:
            if os.path.exists(args.resume):
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch_*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    start_epoch = latest_ckpt
                    _, client_states = model.load_checkpoint(args.resume, tag='epoch_%d' % latest_ckpt) #tag=f"epoch_{completed_epoch}"
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {latest_ckpt})")
                else:
                    logging.info("=> no checkpoint found at '{}'".format(args.resume))
            else:
                logging.info("=> '{}' is not existing!".format(args.resume))
        else:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
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
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))
    
    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model), pin_memory=not args.no_pin_memory)
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        if is_master(args):
            logging.info(f"total_steps: {total_steps}")
        scheduler = warmup_cosine_lr(optimizer, args, total_steps)

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
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
            settings=wandb.Settings(start_method="fork")
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if args.extract_features:
        with torch.no_grad():
            extract_features(model, data, args, device)
        return
        
    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    # torch.cuda.synchronize()
    max_epochs = args.epochs
    # if args.lr_mode:max_epochs=20

    
    args.dataset_dir = '/data/priyank/synthetic/oxford_pets'
    args.dataset = 'pets'
    if not os.path.exists(args.dataset_dir):
        args.dataset_dir = None
        args.dataset = 'imagenet_sketch'
    
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    test = create_dataset_zero_shot(args.dataset, dtd_split=1, low_resolution=args.low_resolution,\
        org_resolution=336 if args.model=='EVA02-CLIP-L-14-336' else 224, root=args.dataset_dir, 
    )
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=3)
    zeroshot_weights = zeroshot_classifier(classes, templates, model, args.model) # torch.Size([512, 1000])
    # evaluate(loader, model, zeroshot_weights, logging, use_spatial_tokens=False)

    BEST_1 = 0
    BEST_5 = 0
    save = True
    
    for epoch in range(start_epoch, max_epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        # train_one_epoch_lr(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        

        ###### OUTDATED
        # Only Spatial Tken
        # train_one_epoch_lr2(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # logging.info("Pseudo Distillation + Text similarity")
        # train_one_epoch_lr6(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # logging.info("ONLY Text similarity")
        # train_one_epoch_lr7(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # logging.info("Text similarity (only HR text)")
        # train_one_epoch_lr8(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # logging.info("Pseudo Distillation + Clasification + Text similarity")
        # train_one_epoch_lr5(model, data, epoch, optimizer, scaler, scheduler, args, writer)

        # logging.info("Pseudo Distillation + Clasification + Normal Triplet Loss + Text similarity (only HD Text)")
        # train_one_epoch_lr11(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # logging.info("Pseudo Distillation + Clasification + Triplet Loss (SEP) + Text similarity (HD + LR)")
        # train_one_epoch_lr12(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # logging.info("Pseudo Distillation + Clasification + Triplet Loss (SEP) + Text similarity (HD)")
        # train_one_epoch_lr13(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        # logging.info("Pseudo Distillation + Clasification + Resolution Distentanglement")
        # train_one_epoch_lr15(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        
        ###### Shortlisted
        if args.train_fn == "MS":
            logging.info("Pseudo Distillation")
            train_one_epoch_lr3(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        elif args.train_fn == "WCL-MS":
            logging.info("Pseudo Distillation + Clasification ")
            train_one_epoch_lr4(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        elif args.train_fn == "WCL-TXHR-MS":
            logging.info("Pseudo Distillation + Clasification + Text similarity (only HD Text)")
            train_one_epoch_lr9(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        elif args.train_fn == "WCL-STR-MS":
            logging.info("Pseudo Distillation + classification + Triplet Loss (SEP Anchors)")
            train_one_epoch_lr10(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        elif args.train_fn == "WCL-NTR-MS":
            logging.info("Pseudo Distillation + classification + Normal Triplet Loss (NO SEP Anchors)")
            train_one_epoch_lr14(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        
        ###### Ablation 
        elif args.train_fn == "SS-MS" or args.train_fn == "SS-LR-MS":
            logging.info("Self-Supervision")
            train_one_epoch_lr16(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        elif args.train_fn == "SS-MS-E2E":
            logging.info("E2E Self-Supervision")
            train_one_epoch_lr17(model, data, epoch, optimizer, scaler, scheduler, args, writer)

        ###### Ablation 
        elif args.train_fn == "SS-OCL":
            logging.info("Self-Supervision using Octuplet Loss")
            train_one_epoch_lr18(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        
        ###### ROBUST 
        elif args.train_fn == "ROBUST":
            logging.info("ROBUST SAM implement")
            train_one_epoch_lr19(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        
        

        completed_epoch = epoch + 1

        if epoch % 2 ==0 :
            top1, top5 = evaluate(loader, model, zeroshot_weights, logging)
            if top5 > BEST_5:
                BEST_5 = top5
                if top1 >= BEST_1:
                    save = True 
            if top1 > BEST_1:
                BEST_1 = top1
                save = True 
            
        # Saving checkpoints.
        # is_master(args) can not be here while using deepspped, otherwise ckpt can not be saved
        if args.save_logs and save:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or ( args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0) and save:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_Best.pt"),
                )
                save = False
            # if args.save_most_recent:
            #     torch.save(
            #         checkpoint_dict,
            #         os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
            #     )

    logging.info(f"Best Top-1 accuracy: {BEST_1:.2f}")
    logging.info(f"Best Top-5 accuracy: {BEST_5:.2f}")

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
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    main(sys.argv[1:])
