
from typing import Any, Dict, Optional, Tuple, Union

import torch 
from open_clip import CLIP 

from open_clip.model import CLIPVisionCfg
from open_clip.transformer import VisionTransformer, LayerNormFp32, LayerNorm, QuickGELU, Transformer, \
    _expand_token

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch.utils.checkpoint import checkpoint


class CLIP_LR(CLIP):
    def __init__( self, vision_cfg=None, embed_dim= 0 , quick_gelu = False, cast_dtype= None, init_logit_scale=0, **kwargs) :
        super().__init__(vision_cfg=vision_cfg, embed_dim=embed_dim, quick_gelu=quick_gelu, cast_dtype=cast_dtype, **kwargs)
            
        del self.visual
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        act_layer = QuickGELU if quick_gelu else nn.GELU

        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        
        self.visual = VisionTransformer_LR(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )


        self.logit_scale_hr_lr = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.lock_text_tower()

    def lock_text_tower(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.token_embedding.requires_grad = False
        self.token_embedding.weight.requires_grad = False    
        self.positional_embedding.requires_grad = False
        
        self.ln_final.requires_grad = False
        self.ln_final.weight.requires_grad = False
        self.ln_final.bias.requires_grad = False
        

        self.text_projection.requires_grad = False
        self.logit_scale.requires_grad = False
        
        # self.attn_mask.requires_grad = False
        # self.logit_bias.requires_grad= False
    
    def encode_image(self, image, normalize: bool = False, **kwargs):
        features = self.visual(image, **kwargs)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image = None, text=None, use_spatial_tokens=True):
        
        text_features = None 
        image_features = self.encode_image(image, normalize=True, use_spatial_tokens=use_spatial_tokens) 
        if text is not None:
            text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale_hr_lr.exp()

        
        





class Transformer_LR(Transformer):
    def __init__( self, width=None, layers=0, num_patches=0, **kawrgs):
        super().__init__(width=width, layers=layers, **kawrgs)
        
        self.lock_existing_Layers()

        self.blocks_spatial_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, num_patches, width))
        for i in range(layers)
        ])
        for i in range(layers):
            nn.init.normal_(self.blocks_spatial_tokens[i], std=.02)
        
    def lock_existing_Layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def add_spatial_tokens(self, x, i):
        cltokens, x = x[:, 0] , x[:, 1:]
        spatial_tokens = self.blocks_spatial_tokens[i]
        x = x + spatial_tokens
        x = torch.cat((cltokens.unsqueeze(1), x), dim=1)
        return x 

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, use_spatial_tokens=None):
        for i,r in enumerate(self.resblocks):
            if use_spatial_tokens:
                x = self.add_spatial_tokens(x, i)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x



class VisionTransformer_LR(VisionTransformer):
    def __init__(self, lock=True , layers =0, width=0, heads=0, mlp_ratio=None, act_layer= nn.GELU, ls_init_value=None, norm_layer=LayerNorm, **kwargs):
        super().__init__(layers=layers, width=width, heads=heads, mlp_ratio=mlp_ratio, act_layer=act_layer, ls_init_value=ls_init_value, norm_layer=norm_layer, **kwargs)
        del self.transformer
        if lock:
            import logging
            logging.info("LOCKING THE VISUAL ENCODER .....'")
            self.lock()
        self.depth = layers

        self.spatial_tokens = nn.Parameter(torch.rand(1, self.grid_size[0] * self.grid_size[1], width))
        self.transformer = Transformer_LR(width=width, layers=layers, heads=heads, mlp_ratio=mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer, num_patches=self.grid_size[0] * self.grid_size[1])
        nn.init.normal_(self.spatial_tokens, std=0.01)
        self.init_parameters()

    def forward(self, x: torch.Tensor, use_spatial_tokens=True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        if use_spatial_tokens:
            x += self.spatial_tokens
        
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x, use_spatial_tokens=use_spatial_tokens)

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled

