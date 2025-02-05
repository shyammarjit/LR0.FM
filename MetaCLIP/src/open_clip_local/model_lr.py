import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


from .model import CLIP, VisualTransformer_LR, QuickGELU, CLIPVisionCfg


class CLIP_LR(CLIP):
    def __init__( self,  embed_dim: int, vision_cfg, text_cfg, quick_gelu: bool = False,):
        super().__init__(embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg, quick_gelu=quick_gelu)
        
        
        del self.visual
        vision_cfg = CLIPVisionCfg(**vision_cfg)

        act_layer = QuickGELU if quick_gelu else nn.GELU
        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisualTransformer_LR(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=embed_dim,
            act_layer=act_layer,
        )

        self.logit_scale_hr_lr = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.lock_text_tower()
        self.init_parameters()

    def lock_text_tower(self):
        self.token_embedding.requires_grad = False
        self.token_embedding.weight.requires_grad = False
    
        self.positional_embedding.requires_grad = False
        
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.attn_mask.requires_grad = False
        self.ln_final.requires_grad = False
        self.ln_final.weight.requires_grad = False
        self.ln_final.bias.requires_grad = False
        
        self.text_projection.requires_grad = False
        self.logit_scale.requires_grad = False
        
        
    def encode_image(self, image, **kwargs):
        return self.visual(image, **kwargs)

    def forward(self, image, text=None, clamp_logit_scale_to=None, use_spatial_tokens=True):
        image_features = self.encode_image(image, use_spatial_tokens=use_spatial_tokens)
        image_features = F.normalize(image_features, dim=-1)
        
        if text is not None:
            text_features = self.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
        else:
            text_features = None

        # if clamp_logit_scale_to is not None:
        #     with torch.no_grad():
        #         self.logit_scale.data.clamp_(0, clamp_logit_scale_to)
        return image_features, text_features, self.logit_scale_hr_lr.exp()

