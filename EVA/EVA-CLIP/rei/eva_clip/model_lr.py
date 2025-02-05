

import torch
import torch.nn.functional as F
from torch import nn
from .model import CustomCLIP, _build_vision_tower
import numpy as np 


class CustomCLIP_lr(CustomCLIP):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: None,
            text_cfg: None,
            quick_gelu: bool = False,
            cast_dtype  = None,
            itm_task: bool = False,
            head = True, train_fn=None,
            n_classes=7000, 
            model_mode=None ):
        super().__init__(embed_dim=embed_dim, vision_cfg=vision_cfg, text_cfg=text_cfg, quick_gelu=quick_gelu, cast_dtype=cast_dtype, itm_task=itm_task)

        self.lock_text_tower()
        self.logit_scale.requires_grad = False 
        
        self.e2e = None
        if model_mode:
            if model_mode == -1 :
                self.e2e = True 
                _ = 0 # do nothing training vanilla transofmers without LR tokens 
                self.lr_head = nn.Identity()
                self.lock_image_tower(fine_tuning=True )
            elif model_mode == 2 :
                self.e2e = True 
                del self.visual
                self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype, lr_mode=True, lock=False)
                self.lr_head = nn.Identity()
                self.lock_image_tower(fine_tuning=True )
            elif model_mode in [4, 5, 8, 11, 14, 12]:
                del self.visual
                self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype, lr_mode=True, n_token_layer=model_mode)
                self.lr_head = nn.Linear(embed_dim, n_classes) if head else nn.Identity()
            elif model_mode == -2:
                del self.visual
                self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype, lr_mode=True, vpt_mode=True)
                self.lr_head = nn.Linear(embed_dim, n_classes) if head else nn.Identity()
            else:
                import pdb
                pdb.set_trace()
        else:
            del self.visual
            self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype, lr_mode=True)
            self.lr_head = nn.Linear(embed_dim, n_classes) if head else nn.Identity()
        
        self.logit_scale_bridge = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_lr = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if (train_fn == 'WCL-MS' or train_fn == 'WCL-NTR-MS' or train_fn == 'WCL-STR-MS') or \
            (train_fn == "SS-LR-MS" or train_fn == "MS" or (train_fn == "SS-MS-E2E")  \
            or (train_fn == "SS-OCL")):
            self.logit_scale_bridge.requires_grad = False 
        if train_fn == "MS" or train_fn == "SS-OCL":
            self.lr_head = nn.Identity()

        if train_fn == "SS-OCL":
            self.logit_scale_lr.requires_grad = False
            
        print(" USING LR MODE MODEL .... ")

    def lock_text_tower(self, unlocked_layers:int=0, freeze_layer_norm:bool=True):
        for param in self.text.parameters():
            param.requires_grad = False
        return 

    def lock_image_tower(self, unlocked_layers:int=0, freeze_layer_norm:bool=True, fine_tuning=False):
        if fine_tuning is False:
            assert False, "Locking Vision Backbone not yet verified...."
        else:
            N_BLOCKS = len(self.visual.blocks)
            for name,param in self.visual.named_parameters():
                # if "pos_embed" in name or "patch_embed" in name:
                #     param.requires_grad = False
                #     continue 
                if "blocks" in name and "blocks_" not in name:
                    splits = name.split(".")
                    assert splits [0] == "blocks"
                    block_no = int(splits[1])
                    if block_no < N_BLOCKS * 3 // 4:
                        param.requires_grad = False
                    else:
                        print(name)
                    continue 
                print(name)
            return 


    def encode_image(self, image, normalize: bool = False, **kwargs):
        features = self.visual(image, **kwargs)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text, **kwargs):
        image_features = self.encode_image(image, normalize=True, **kwargs)
        text_features = None
        if text is not None:
            text_features = self.encode_text(text, normalize=True)
        lr_labels = self.lr_head(image_features)
        if self.e2e:
            return image_features, text_features, self.logit_scale_lr.exp(), lr_labels, self.logit_scale_bridge.exp(), self.logit_scale.exp()
        return image_features, text_features, self.logit_scale_lr.exp(), lr_labels, self.logit_scale_bridge.exp()



