import json
import logging
import os
import pathlib
import re

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from functools import partial
from dataclasses import asdict
import torch


from open_clip.model import CLIP, convert_weights_to_lp, get_cast_dtype, set_model_preprocess_cfg
from open_clip.openai import load_openai_model
from open_clip.pretrained import get_pretrained_cfg, download_pretrained, list_pretrained_tags_by_model, download_pretrained_from_hf

from open_clip.factory import load_checkpoint, get_model_config, list_models, _get_hf_config
from open_clip.transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs

from .model_lr import CLIP_LR
HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        lr_mode=  None,
        strict=True,
        **model_kwargs,
):
    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = asdict(PreprocessCfg())
    has_hf_hub_prefix = model_name.startswith(HF_HUB_PREFIX)
    if has_hf_hub_prefix:
        model_id = model_name[len(HF_HUB_PREFIX):]
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
        config = _get_hf_config(model_id, cache_dir)
        preprocess_cfg = merge_preprocess_dict(preprocess_cfg, config['preprocess_cfg'])
        model_cfg = config['model_cfg']
        pretrained_hf = False  # override, no need to load original HF text weights
    else:
        model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
        checkpoint_path = None
        model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(
            model_name,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            logging.info(f'Loaded {model_name} model config.')
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size

        is_timm_model = 'timm_model_name' in model_cfg.get('vision_cfg', {})
        if pretrained_image:
            if is_timm_model:
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        # cast_dtype set for fp16 and bf16 (manual mixed-precision), not set for 'amp' or 'pure' modes
        cast_dtype = get_cast_dtype(precision)
        is_hf_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
        if is_hf_model:
            # load pretrained weights for HF text model IFF no CLIP weights being loaded
            model_cfg['text_cfg']['hf_model_pretrained'] = pretrained_hf and not pretrained
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_model

        model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)
        if lr_mode:
            model = CLIP_LR(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        if precision in ("fp16", "bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            # manual mixed precision that matches original OpenAI behaviour
            if is_timm_model:
                # FIXME this is a bit janky, create timm based model in low-precision and
                # then cast only LayerNormFp32 instances back to float32 so they don't break.
                # Why? The convert_weights_to_lp fn only works with native models.
                model.to(device=device, dtype=dtype)
                from .transformer import LayerNormFp32

                def _convert_ln(m):
                    if isinstance(m, LayerNormFp32):
                        m.weight.data = m.weight.data.to(torch.float32)
                        m.bias.data = m.bias.data.to(torch.float32)
                model.apply(_convert_ln)
            else:
                model.to(device=device)
                convert_weights_to_lp(model, dtype=dtype)
        elif precision in ("pure_fp16", "pure_bf16"):
            dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
            model.to(device=device, dtype=dtype)
        else:
            model.to(device=device)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = ''
            pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
            if pretrained_cfg:
                checkpoint_path = download_pretrained(pretrained_cfg, cache_dir=cache_dir)
                preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                error_str = (
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    f' Available pretrained tags ({list_pretrained_tags_by_model(model_name)}.')
                logging.warning(error_str)
                raise RuntimeError(error_str)
            pretrained_loaded = True
        elif has_hf_hub_prefix:
            logging.info(f'Loading pretrained {model_name} weights ({checkpoint_path}).')
            load_checkpoint(model, checkpoint_path, strict=strict)
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    # set image preprocessing configuration in model attributes for convenience
    if getattr(model.visual, 'image_size', None) is not None:
        # use image_size set on model creation (via config or force_image_size arg)
        force_preprocess_cfg['size'] = model.visual.image_size
    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))

    return model



def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val



def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {}, mean=image_mean, std=image_std, interpolation=image_interpolation, resize_mode=image_resize_mode)

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        require_pretrained=True,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess