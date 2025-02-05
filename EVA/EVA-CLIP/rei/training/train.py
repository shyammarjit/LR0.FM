import json
import logging
import math
import os
import time
import gc

import numpy as np
import torch
import torch.nn as nn
from torch import inf
import torch.nn.functional as F
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None
from eva_clip import ClipLoss, get_cast_dtype, get_tokenizer
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast
from .utils import save_file
from torchvision.utils import save_image
from einops import rearrange, repeat
from eva_clip.loss import feat_lr
from .triplet_loss import TripletLoss, TripletLoss_w_anchors
from collections import defaultdict 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.to(dtype=torch.float32)

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        )


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss, acc = loss(image_features, text_features, logit_scale)
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def extract_features(model, data, args, device):
    
    img_emb_folder = args.img_emb_path
    text_emb_folder = args.text_emb_path

    save_interval = args.save_interval if args.save_interval else 100
    all_features = []
    feature_info = {}

    model.eval()
    # autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if 'val' in data:
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples
        
        all_image_features = []
        all_text_features = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                idx = i+1

                images, texts = batch

                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                image_features, text_features = model(images, texts)

                all_image_features.append(image_features)
                all_text_features.append(text_features)

                batch_size = images.shape[0]
                num_samples += batch_size
                logging.info(
                    f"Extract RANK: {args.rank} [{num_samples} / {samples_per_val}]"
                )

                if idx % save_interval == 0:

                    img_feat = np.concatenate(all_image_features)
                    text_feat = np.concatenate(all_text_features)
                    

                    split = "%08d" % (idx//save_interval)
                    out_img_feat_file = (
                        f"{img_emb_folder}/rank{args.rank}_img_emb_{split}.npy"
                    )
                    out_text_feat_file = (
                        f"{text_emb_folder}/rank{args.rank}_text_emb_{split}.npy"
                    )

                    save_file(img_feat, out_img_feat_file)
                    save_file(text_feat, out_text_feat_file)

                    
                    all_image_features = []
                    all_text_features = []

            if len(all_image_features) > 0:
                img_feat = np.concatenate(all_image_features)
                text_feat = np.concatenate(all_text_features)

                split = "%08d" % ((idx//save_interval)+1)
                out_img_feat_file = (
                    f"{img_emb_folder}/rank{args.rank}_img_emb_{split}.npy"
                )
                out_text_feat_file = (
                    f"{text_emb_folder}/rank{args.rank}_text_emb_{split}.npy"
                )

                save_file(img_feat, out_img_feat_file)
                save_file(text_feat, out_text_feat_file)
    torch.distributed.barrier()

def train_one_epoch_lr(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, _ = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        images = rearrange(images, "B N ... -> (B N) ...")
        # save_image(images, "images.png")
        # images = rearrange(images, "(B N) ... -> B N ...", B=BATCH, N=2)
        # image_hr = images[:,0]
        # image_lr = images[:,1]
        # save_image(image_hr, "hr.png"), save_image(image_lr, "lr.png")
        
        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with autocast():
            image_features, _, logit_scale = model(images, None)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            image_features_hr = image_features[:,0]
            image_features_lr = image_features[:,1]

            total_loss, acc = loss(image_features_hr, image_features_lr, logit_scale)
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Only Spatial Tken
def train_one_epoch_lr2(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, _ = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        
        images = rearrange(images, "B N ... -> (B N) ...")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with autocast():
            # image_features_HR, _, logit_scale = model(IMG_HR, None)
            # image_features_LR, _, logit_scale = model(IMG_LR, None)
            image_features, _, logit_scale = model(images, None)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            total_loss, acc = loss(image_features_HR, image_features_LR, logit_scale)
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()


            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation 
def train_one_epoch_lr3(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, text = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        IMG_HR = images[:,0] 
        N_LR = images.shape[1] -1
        
        images = rearrange(images, "B N ... -> (B N) ...")
        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, _, logit_scale, _, _ = model(images, None)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N= N_LR + 1)
            image_features_HR = image_features[:,0]
            
            # acc = defaultdict(int)
            total_loss_hr, acc = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            if N_LR == 1:
                image_features_LR = image_features[:,1]
                total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)    
            else:
                image_features_LR = image_features[:,1:]
                total_loss_lr = 0
                for i in range(N_LR):
                    loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR[:,i], logit_scale)
                    total_loss_lr += loss_lr
                
            total_loss = total_loss_lr + total_loss_hr
            clip_loss = total_loss.clone().detach()
            
        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        scaler.scale(total_loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + classification
def train_one_epoch_lr4(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0    
    # print("*****", len(dataloader))
    # 21000 , 21000
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, text = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        # N = images.shape[1]
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= N_LR + 1).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()
        
        # print(IMG_HR.shape, IMG_LR.shape, images.shape)
        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, _, logit_scale, lr_label_pred, _ = model(images, None)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N= N_LR + 1)
            image_features_HR = image_features[:,0]
            

            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)

            # acc = defaultdict(int)
            total_loss_hr, acc = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            if N_LR == 1:
                image_features_LR = image_features[:,1]
                total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)    
                # for key in acc_lr:
                #     acc[key] = (acc_lr[key] + acc[key]) / 2
            else:
                image_features_LR = image_features[:,1:]
                total_loss_lr = 0
                
                for i in range(N_LR):
                    loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR[:,i], logit_scale)
                    total_loss_lr += loss_lr
                    # for key in acc_lr:acc[key] += acc_lr[key]
                # for key in acc_lr:acc[key] = (acc[key] + acc_hr[key]) / (N_LR + 1)
                
            total_loss = total_loss_lr + total_loss_hr + ce_loss
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        
        scaler.scale(total_loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
        
            loss_scale_value = scaler.get_scale()
            grad_nrom = get_grad_norm_(model.parameters())
            
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    

# Pseudo Distillation + classification + Text similarity 
def train_one_epoch_lr5(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")
        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, text_features_HR_Truth, logit_scale, _, _ = model(IMG_HR, texts, use_spatial_tokens=False)

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge = model(images, texts)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N=2)

            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]

            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)

            acc = {}
            total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            total_loss = total_loss_lr + total_loss_hr + ce_loss
            for key in acc_lr:
                acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            
            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
            total_loss_hr_lr, acc_hr_lr = text_loss(image_features_HR, text_features_LR, logit_scale_bridge)
            total_loss_lr_lr, acc_lr_lr = text_loss(image_features_LR, text_features_LR, logit_scale_bridge)

            total_loss += total_loss_hr_hr + total_loss_hr_lr + total_loss_lr_hr + total_loss_lr_lr 

            for key in acc_hr_hr:acc[key + "_bridge"] = (acc_hr_hr[key] + acc_lr_hr[key] + acc_hr_lr[key] + acc_lr_lr[key]) / 4
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + Text similarity 
def train_one_epoch_lr6(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")
        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, text_features_HR_Truth, logit_scale, _, _ = model(IMG_HR, texts, use_spatial_tokens=False)

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge = model(images, texts)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N=2)

            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]

            acc = {}
            total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            total_loss = total_loss_lr + total_loss_hr
            for key in acc_lr:
                acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            
            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
            total_loss_hr_lr, acc_hr_lr = text_loss(image_features_HR, text_features_LR, logit_scale_bridge)
            total_loss_lr_lr, acc_lr_lr = text_loss(image_features_LR, text_features_LR, logit_scale_bridge)

            total_loss += total_loss_hr_hr + total_loss_hr_lr + total_loss_lr_hr + total_loss_lr_lr 

            for key in acc_hr_hr:acc[key + "_bridge"] = (acc_hr_hr[key] + acc_lr_hr[key] + acc_hr_lr[key] + acc_lr_lr[key]) / 4
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Text similarity 
def train_one_epoch_lr7(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")
        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge = model(images, texts)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N=2)

            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]

            acc = {}
            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
            total_loss_hr_lr, acc_hr_lr = text_loss(image_features_HR, text_features_LR, logit_scale_bridge)
            total_loss_lr_lr, acc_lr_lr = text_loss(image_features_LR, text_features_LR, logit_scale_bridge)

            total_loss = total_loss_hr_hr + total_loss_hr_lr + total_loss_lr_hr + total_loss_lr_lr 

            for key in acc_hr_hr:acc[key] = (acc_hr_hr[key] + acc_lr_hr[key] + acc_hr_lr[key] + acc_lr_lr[key]) / 4
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Text similarity (only HR text)
def train_one_epoch_lr8(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")
        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge = model(images, texts)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N=2)

            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]

            acc = {}
            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
            
            total_loss = total_loss_hr_hr + total_loss_lr_hr

            for key in acc_hr_hr:acc[key] = (acc_hr_hr[key] + acc_lr_hr[key]) / 2
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + classification + Text similarity + ONLY HD Text 
def train_one_epoch_lr9(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= N_LR + 1).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")
        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, text_features_HR_Truth, logit_scale, _, _ = model(IMG_HR, texts, use_spatial_tokens=False)

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge = model(images, texts)
            
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=N_LR + 1)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N= 2)

            image_features_HR = image_features[:,0]
            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]
            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)

            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            acc = defaultdict(int)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)

            if N_LR == 1:
                image_features_LR = image_features[:,1]
                total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
                total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
                for key in acc_lr:
                    acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            else:
                image_features_LR = image_features[:,1:]
                total_loss_lr = 0 
                total_loss_lr_hr = 0 
                for i in range(N_LR):
                    loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR[:,i], logit_scale)
                    total_loss_lr += loss_lr
                    loss_lr_hr, acc_lr_hr = text_loss(image_features_LR[:,i], text_features_HR, logit_scale_bridge)
                    total_loss_lr_hr += loss_lr_hr
                    for key in acc_lr:acc[key] += acc_lr[key]
                for key in acc_lr:acc[key] = (acc[key] + acc_hr[key]) / (N_LR + 1)
            
            total_loss = total_loss_lr + total_loss_hr + ce_loss
            total_loss += total_loss_hr_hr + total_loss_lr_hr 

            for key in acc_hr_hr:acc[key + "_bridge"] = (acc_hr_hr[key] + acc_lr_hr[key]) / 2
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + classification + Triplet Loss (SEP Anchors)
def train_one_epoch_lr10(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    # TRI_LOSS = TripletLoss(margin=0.3)
    TRI_LOSS = TripletLoss_w_anchors(margin=0.3, metric='cosine')
    

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label_res = lr_label.to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = repeat(lr_label, "B -> B N ", N= N_LR + 1).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")

        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, _, logit_scale, lr_label_pred, _  = model(images, None)
            
            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)

            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N= N_LR + 1)
            image_features_HR = image_features[:,0]
            triplet_loss = TRI_LOSS(emb1=image_features_HR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            acc = defaultdict(int)

            if N_LR ==1 :
                image_features_LR = image_features[:,1]
                triplet_loss += TRI_LOSS(emb1=image_features_HR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
                triplet_loss += TRI_LOSS(emb1=image_features_LR, emb2=image_features_LR, emb3=image_features_LR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
                triplet_loss += TRI_LOSS(emb1=image_features_LR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)

                total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale) 
                # total_loss_lr.item(), total_loss_hr.item(), ce_loss.item(), triplet_loss.item()
                for key in acc_lr:
                    acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            else:
                image_features_LR = image_features[:,1:]
                total_loss_lr = 0 
                for i in range(N_LR):
                    triplet_loss += TRI_LOSS(emb1=image_features_HR, emb2=image_features_LR[:,i], emb3=image_features_LR[:,i], label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
                    triplet_loss += TRI_LOSS(emb1=image_features_LR[:,i], emb2=image_features_LR[:,i], emb3=image_features_LR[:,i], label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
                    triplet_loss += TRI_LOSS(emb1=image_features_LR[:,i], emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
                    
                    loss_hr, acc_lr = loss(image_features_HR_Truth, image_features_LR[:,i], logit_scale)
                    for key in acc_lr:acc[key] += acc_lr[key]
                    total_loss_lr += loss_hr

                for key in acc_lr: acc[key] = (acc[key] + acc_hr[key]) / (N_LR + 1)
                
            
            total_loss = total_loss_lr + total_loss_hr + ce_loss + triplet_loss
            
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + Clasification + Normal Triplet Loss + Text similarity (only HD Text)
def train_one_epoch_lr11(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    TRI_LOSS = TripletLoss(margin=0.3)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")

        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge  = model(images, texts)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N=2)

            triplet_loss = TRI_LOSS(image_features, lr_label, normalize_feature=True)[0]
            
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]

            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)
            
            acc = {}
            total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            total_loss = total_loss_lr + total_loss_hr + ce_loss + triplet_loss
            for key in acc_lr:
                acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
            
            for key in acc_hr_hr:acc[key + "_bridge"] = (acc_hr_hr[key] + acc_lr_hr[key]) / 2
            total_loss += total_loss_hr_hr + total_loss_lr_hr
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + Clasification + Triplet Loss (SEP) + Text similarity (HD + LR)
def train_one_epoch_lr12(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    # TRI_LOSS = TripletLoss(margin=0.3)
    TRI_LOSS = TripletLoss_w_anchors(margin=0.3, metric='cosine')

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")
        
        lr_label_res = lr_label.to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")

        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge  = model(images, texts)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N=2)

            # triplet_loss = TRI_LOSS(image_features, lr_label, normalize_feature=True)[0]
            
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]

            triplet_loss = TRI_LOSS(emb1=image_features_HR, emb2=image_features_LR, emb3=image_features_LR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
            triplet_loss += TRI_LOSS(emb1=image_features_HR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
            triplet_loss += TRI_LOSS(emb1=image_features_LR, emb2=image_features_LR, emb3=image_features_LR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
            triplet_loss += TRI_LOSS(emb1=image_features_LR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)

            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)
            
            acc = {}
            total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            total_loss = total_loss_lr + total_loss_hr + ce_loss + triplet_loss
            for key in acc_lr:
                acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
            total_loss_hr_lr, acc_hr_lr = text_loss(image_features_HR, text_features_LR, logit_scale_bridge)
            total_loss_lr_lr, acc_lr_lr = text_loss(image_features_LR, text_features_LR, logit_scale_bridge)
            

            for key in acc_hr_hr:acc[key + "_bridge"] = (acc_hr_hr[key] + acc_lr_hr[key] + acc_hr_lr[key] + acc_lr_lr[key]) / 4
            total_loss += total_loss_hr_hr + total_loss_lr_hr + total_loss_hr_lr + total_loss_lr_lr 
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + Clasification + Triplet Loss (SEP) + Text similarity (HD)
def train_one_epoch_lr13(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    text_loss = ClipLoss( local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    # TRI_LOSS = TripletLoss(margin=0.3)
    TRI_LOSS = TripletLoss_w_anchors(margin=0.3, metric='cosine')

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")
        
        lr_label_res = lr_label.to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        texts = rearrange(texts, "B N ... -> (B N) ...")

        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge  = model(images, texts)
            text_features = rearrange(text_features, "(B N) ... -> B N ...", B=BATCH, N=2)

            # triplet_loss = TRI_LOSS(image_features, lr_label, normalize_feature=True)[0]
            
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            text_features_HR = text_features[:,0]
            text_features_LR = text_features[:,1]

            triplet_loss = TRI_LOSS(emb1=image_features_HR, emb2=image_features_LR, emb3=image_features_LR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
            triplet_loss += TRI_LOSS(emb1=image_features_HR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
            triplet_loss += TRI_LOSS(emb1=image_features_LR, emb2=image_features_LR, emb3=image_features_LR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
            triplet_loss += TRI_LOSS(emb1=image_features_LR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)

            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)
            
            acc = {}
            total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            total_loss = total_loss_lr + total_loss_hr + ce_loss + triplet_loss
            for key in acc_lr:
                acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            total_loss_hr_hr, acc_hr_hr = text_loss(image_features_HR, text_features_HR, logit_scale_bridge)
            total_loss_lr_hr, acc_lr_hr = text_loss(image_features_LR, text_features_HR, logit_scale_bridge)
            for key in acc_hr_hr:acc[key + "_bridge"] = (acc_hr_hr[key] + acc_lr_hr[key]) / 2
            total_loss += total_loss_hr_hr + total_loss_lr_hr 
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + classification + Normal Triplet Loss (NO SEP Anchors)
def train_one_epoch_lr14(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    TRI_LOSS = TripletLoss(margin=0.3)
    

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, texts = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label_res = lr_label.to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = repeat(lr_label, "B -> B N ", N= N_LR + 1).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")

        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, _, logit_scale, lr_label_pred, _  = model(images, None)
            triplet_loss = TRI_LOSS(image_features, lr_label, normalize_feature=True)[0]
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N= N_LR + 1)
            image_features_HR = image_features[:,0]
            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)

            acc = defaultdict(int)
            if N_LR == 1:
                image_features_LR = image_features[:,1]
                total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
                # total_loss_lr.item(), total_loss_hr.item(), ce_loss.item(), triplet_loss.item()
                for key in acc_lr:
                    acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            else:
                image_features_LR = image_features[:,1:]
                total_loss_lr = 0 
                for i in range(N_LR):
                    loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR[:,i], logit_scale)
                    total_loss_lr += loss_lr
                    for key in acc_lr:acc[key] += acc_lr[key]
                for key in acc_lr:acc[key] = (acc[key] + acc_hr[key]) / (N_LR + 1)
            
            total_loss = total_loss_lr + total_loss_hr + ce_loss + triplet_loss
            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

# Pseudo Distillation + classification + Resolution Distentanglment (No Action Loss)
def train_one_epoch_lr15(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, text = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        text = text.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= 2).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")

        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, _, logit_scale, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features, _, logit_scale, lr_label_pred, _ = model(images, None)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N=2)
            image_features_HR = image_features[:,0]
            image_features_LR = image_features[:,1]

            # save_image(images, "temp.png")
            ce_loss = F.cross_entropy(lr_label_pred, lr_label)

            acc = {}
            total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)
            total_loss_hr, acc_hr = loss(image_features_HR_Truth, image_features_HR, logit_scale)
            total_loss = total_loss_lr + total_loss_hr + ce_loss
            for key in acc_lr:
                acc[key] = (acc_lr[key] + acc_hr[key]) / 2
            
            import pdb
            pdb.set_trace()

            
            # total_loss += F.kl_div( F.log_softmax(logit_scale_LR / tau, dim=-1), F.log_softmax(logit_scale_HR / tau, dim=-1), reduction='sum', log_target=True ) * (tau * tau) / logit_scale_LR.numel()

            clip_loss = total_loss.clone().detach()

        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            
            scaler.step(optimizer)
            scaler.update()
        elif args.enable_deepspeed:
            model.backward(total_loss)
            model.step()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for




# Self supervision 
def train_one_epoch_lr16(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0    
    # print("*****", len(dataloader))
    # 21000 , 21000
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, text = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        # N = images.shape[1]
        IMG_HR = images[:,0] 
        # save_image(IMG_HR, "images_HR.png")
        IMG_LR = images[:,1:] 
        N_LR = IMG_LR.shape[1]
        IMG_LR = rearrange(IMG_LR, "B N ... -> (B N) ...")
        # save_image(IMG_LR, "images_LR.png")

        lr_label = repeat(lr_label, "B -> B N ", N= N_LR + 1).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = rearrange(lr_label, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        # save_image(images, "images.png")

        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()
        
        
        with autocast():
            image_features, _, logit_scale, lr_label_pred, _ = model(images, None)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N= N_LR + 1)
            image_features_HR = image_features[:,0]
            

            if N_LR == 1:
                image_features_LR = image_features[:,1]
                total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)    
                # for key in acc_lr:
                #     acc[key] = (acc_lr[key] + acc[key]) / 2
            else:
                image_features_LR = image_features[:,1:]
                total_loss_lr = 0
                
                for i in range(N_LR):
                    loss_lr, acc_lr = loss(image_features_HR, image_features_LR[:,i], logit_scale)
                    total_loss_lr += loss_lr
                    # for key in acc_lr:acc[key] += acc_lr[key]
                # for key in acc_lr:acc[key] = (acc[key] + acc_hr[key]) / (N_LR + 1)
                
            total_loss = total_loss_lr
            clip_loss = total_loss.clone().detach()
        
        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        
        scaler.scale(total_loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
        
            loss_scale_value = scaler.get_scale()
            grad_nrom = get_grad_norm_(model.parameters())
            
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    

# E2E - Self supervision 
def train_one_epoch_lr17(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss_text = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0    
    # print("*****", len(dataloader))
    # 21000 , 21000
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, text = batch

        text = text[:,0]
        text = text.to(device=device, dtype=cast_dtype, non_blocking=True)
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        
        

        BATCH = images.shape[0]
        N_LR = images.shape[1] - 1
        images = rearrange(images, "B N ... -> (B N) ...")
        
        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()
        
        with autocast():
            image_features, text_features, logit_scale, lr_label_pred, logit_scale_bridge, logit_scale_orig = model(images, text)
            
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N= N_LR + 1)
            image_features_HR = image_features[:,0]

            total_loss_text, _ = loss_text(image_features_HR, text_features, logit_scale_orig)
            if N_LR == 1:
                image_features_LR = image_features[:,1]
                total_loss_lr, acc_lr = loss(image_features_HR_Truth, image_features_LR, logit_scale)    
                # for key in acc_lr:
                #     acc[key] = (acc_lr[key] + acc[key]) / 2
                loss_text_lr, _ = loss_text(image_features_LR, text_features, logit_scale_orig)
                total_loss_text += loss_text_lr
            else:
                image_features_LR = image_features[:,1:]
                total_loss_lr = 0
                
                for i in range(N_LR):
                    loss_lr, acc_lr = loss(image_features_HR, image_features_LR[:,i], logit_scale)
                    total_loss_lr += loss_lr

                    loss_text_lr, _ = loss_text(image_features_LR[:,i], text_features, logit_scale_orig)
                    total_loss_text += loss_text_lr

                    # for key in acc_lr:acc[key] += acc_lr[key]
                # for key in acc_lr:acc[key] = (acc[key] + acc_hr[key]) / (N_LR + 1)
            
            total_loss = total_loss_lr + total_loss_text
            clip_loss = total_loss.clone().detach()
        
        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        
        scaler.scale(total_loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
        
            loss_scale_value = scaler.get_scale()
            grad_nrom = get_grad_norm_(model.parameters())
            
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    


# Distillation using OCTET LOSS
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10042669
def train_one_epoch_lr18(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    TRI_LOSS = TripletLoss_w_anchors(margin=0.3, metric='cosine')


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, text = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label = lr_label.to(device=device, dtype=cast_dtype, non_blocking=True)

        BATCH = images.shape[0]
        IMG_HR = images[:,0] 
        N_LR = images.shape[1] -1
        
        lr_label_res = repeat(lr_label, "B -> B N ", N= N_LR).to(device=device, dtype=cast_dtype, non_blocking=True)
        lr_label_res = rearrange(lr_label_res, "B N ... -> (B N) ...")
        
        images = rearrange(images, "B N ... -> (B N) ...")
        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with autocast():
            image_features, _, logit_scale, _, _ = model(images, None)
            image_features = rearrange(image_features, "(B N) ... -> B N ...", B=BATCH, N= N_LR + 1)
            image_features_HR = image_features[:,0]
            
            triplet_loss = TRI_LOSS(emb1=image_features_HR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label, label2=lr_label, label3=lr_label)
            if N_LR == 1:
                image_features_LR = image_features[:,1]
                assert False, "Not yet verified Octuplet Loss for N_LR == 1"
            else:
                image_features_LR_16 = image_features[:,1]
                image_features_LR = image_features[:,1:]
                image_features_LR = rearrange(image_features_LR, "B N ... -> (B N) ...")

                logits_per_image = image_features_HR @ image_features_LR_16.T
                logits_per_text = image_features_LR_16 @ image_features_HR.T

                labels = torch.arange(BATCH, device=device, dtype=torch.long)
                i2t_acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
                t2i_acc = (logits_per_text.argmax(-1) == labels).sum() / len(logits_per_text)
                acc = {"i2t": i2t_acc, "t2i": t2i_acc}

            triplet_loss  += TRI_LOSS(emb1=image_features_HR, emb2=image_features_LR, emb3=image_features_LR, label1=lr_label, label2=lr_label_res, label3=lr_label_res)
            triplet_loss  += TRI_LOSS(emb1=image_features_LR, emb2=image_features_HR, emb3=image_features_HR, label1=lr_label_res, label2=lr_label, label3=lr_label)
            triplet_loss  += TRI_LOSS(emb1=image_features_LR, emb2=image_features_LR, emb3=image_features_LR, label1=lr_label_res, label2=lr_label_res, label3=lr_label_res)
    
            total_loss = triplet_loss
            clip_loss = total_loss.clone().detach()
            
        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        scaler.scale(total_loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



# ROBUST SAM
def train_one_epoch_lr19(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    # loss = feat_lr(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)
    loss = ClipLoss(local_loss=args.local_loss, gather_with_grad=args.gather_with_grad, cache_labels=True, rank=args.rank, world_size=args.world_size,)


    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_scaler = AverageMeter()
    grad_norm_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # for param in model.module.visual.parameters():print(param.shape, param.requires_grad)
    tau = 0.8
    accumulate_count = 0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, lr_label, text = batch

        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        BATCH = images.shape[0]
        IMG_HR = images[:,0] 
        IMG_LR = images[:,1:] 
        N_LR = images.shape[1] -1
        
        images = rearrange(images, "B N ... -> (B N) ...")
        IMG_LR = rearrange(IMG_LR , "B N ... -> (B N) ...") 
        data_time_m.update(time.time() - end)
        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        with torch.no_grad():
            image_features_HR_Truth, robust_tokens_HR_Truth, _, _, _ = model(IMG_HR, None, use_spatial_tokens=False)

        with autocast():
            image_features_LR, robust_tokens_LR, logit_scale, _, _ = model(IMG_LR, None)
            
            image_features_LR = rearrange(image_features_LR, "(B N) ... -> B N ...", B=BATCH, N= N_LR)
            robust_tokens_LR = rearrange(robust_tokens_LR, "(B N) ... -> B N ...", B=BATCH, N= N_LR)

            Mask_Feature_Consistency_Loss = 0
            token_consistency_loss = 0 
            for i in range(N_LR):
                diff = robust_tokens_LR[:,i] - robust_tokens_HR_Truth
                diff = (diff ** 2).mean()
                Mask_Feature_Consistency_Loss += diff

                diff = image_features_LR[:,i] - image_features_HR_Truth
                diff = (diff ** 2).sum(-1).mean()
                token_consistency_loss += diff

                image_features_LR_16 = image_features_LR[:,0]
                logits_per_image = image_features_HR_Truth @ image_features_LR_16.T
                logits_per_text = image_features_LR_16 @ image_features_HR_Truth.T

                labels = torch.arange(BATCH, device=device, dtype=torch.long)
                i2t_acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
                t2i_acc = (logits_per_text.argmax(-1) == labels).sum() / len(logits_per_text)
                acc = {"i2t": i2t_acc, "t2i": t2i_acc}
                
            total_loss = token_consistency_loss + Mask_Feature_Consistency_Loss
            clip_loss = total_loss.clone().detach()
            
        loss_list = [torch.zeros_like(total_loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, total_loss)
        loss_list = torch.tensor(loss_list)

        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()
        if loss_list_isnan or loss_list_isinf:
            logging.info(f" ==================== loss_isnan = {loss_list_isnan},  loss_isinf = {loss_list_isinf} ==================== ")

        scaler.scale(total_loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        
        if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            loss_clip_m.update(clip_loss.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            if args.enable_deepspeed:
                loss_scale_value, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale_value = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale_value = 0.0
                grad_nrom = get_grad_norm_(model.parameters())
            loss_scaler.update(loss_scale_value, batch_size)
            grad_norm_m.update(grad_nrom, batch_size)

            index_visual, index_text = 0, 0
            for i, v in enumerate(optimizer.param_groups):
                if v['group'] == 'visual' and v['lr_scale'] == 1.0:
                    index_visual = i
                if v['group'] == 'text' and v['lr_scale'] == 1.0:
                    index_text = i

            logging.info(
                f"Global Steps: {step + 1} "
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss(CLIP): {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                f"Grad Norm: {grad_norm_m.val:#.5g} ({grad_norm_m.avg:#.4g}) "
                f"Loss Scaler: {loss_scaler.val:#.5g} ({loss_scaler.avg:#.4g}) "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"LR_visual: {optimizer.param_groups[index_visual]['lr']:5f} "
                f"LR_text: {optimizer.param_groups[index_text]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} "
                f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                f"t2i_acc: {acc['t2i'].item() * 100:.2f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s"
            )
            
            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_clip": loss_clip_m.val,
                "loss_scaler": loss_scaler.val,
                "grad_nrom": grad_norm_m.val,
                "i2t_acc": acc['i2t'].item() * 100,
                "t2i_acc": acc['t2i'].item() * 100,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_visual": optimizer.param_groups[index_visual]["lr"],
                "lr_text": optimizer.param_groups[index_text]["lr"],
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



