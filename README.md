<div align="center">

## 🚀 LR0.FM (ICLR-25 🎉)<br> [webpage](https://ucf-crcv.github.io/lr0.fm/) | [paper](https://arxiv.org/abs/2502.03950) | [video](https://recorder-v3.slideslive.com/#/share?share=99927&s=b52e48b7-e501-45c7-b7c9-b1d415e77f1e) | [results]() | [weights]()<br><br> <p align="left">💡 Highlights</p>
</div>
✨ We introduce LR0.FM, a comprehensive benchmark evaluating the impact of low resolution on the zero-shot classification performance of 10 FM(s) across 66 backbones and 15 datasets.<br/>
✨ We propose a novel metric, Weighted Aggregated Robustness, to address the limitations of existing metrics and better evaluate model performance across resolutions and datasets.<br/>
✨ Our key findings show that: (i) model size positively correlates with robustness to resolution degradation, (ii) pre-training dataset quality is more important than its size, and (iii) fine-tuned and higher resolution models are less robust against LR.<br/>
✨ Our analysis further reveals that the model makes semantically reasonable predictions at LR, and the lack of fine-grained details in input adversely impacts the model’s initial layers more than the deeper layers.<br/>
✨ Our proposed LR-TK0 enhances model robustness to low-resolution without altering pre-trained weights, demonstrating effectiveness across several datasets and its generalization capability across backbones and other approaches.<br/>
</div>


## Zero-Shot Classification Results in Low Resolution

Model | Backbones | Results | Code |
| --- | --- | --- | --- |
|[CLIP](https://arxiv.org/pdf/2103.00020)|CLIP-ViT-B/32, CLIP-ViT-B/16, CLIP-ViT-L/14, CLIP-ViT-L/14@336px, CLIP-RN50, CLIP-RN101, CLIP-RN50x4, CLIP-RN50x16, CLIP-RN50x64|[sheet](https://docs.google.com/spreadsheets/d/12Pl9B43kzD_OuOvlQc7vs7V8e49LuWq4AJhftX_AusM/edit?usp=sharing)|[link](https://github.com/openai/CLIP)|
|[BLIP](https://arxiv.org/pdf/2201.12086)|BLIP-ViT-B/16 (14M), BLIP-ViT-B/16 (129M), BLIP-ViT-B/16 & CapFilt-L (129M), BLIP-ViT-L/16 (129M), BLIP-ViT-B/16 (129M + COCO), BLIP-ViT-B/16 (129M + Flickr), BLIP-ViT-L/16 (129M + COCO), BLIP-ViT-L/16 (129M + Flickr)|[sheet](https://docs.google.com/spreadsheets/d/12idCO5Fv90XDixAgi25elp7JEU9DgLiKJ_ZKWrCpPdk/edit?usp=sharing)|[link](https://github.com/salesforce/BLIP)|
|[MetaCLIP](https://arxiv.org/pdf/2309.16671)|MetaCLIP-ViT-B/32 (400M), MetaCLIP-ViT-B/32 (2.5B), MetaCLIP-ViT-B/16 (400M), MetaCLIP-ViT-B/16 (2.5B), MetaCLIP-ViT-L/14 (400M), MetaCLIP-ViT-L/14 (2.5B), MetaCLIP-ViT-H/14 (2.5B), MetaCLIP-ViT-G/14 (2.5B)|[sheet](https://docs.google.com/spreadsheets/d/1eyJD5P8DA22bclCZyk0C9iFbSmkvVVYMy-dRLfJIgoY/edit?usp=sharing)|[link](https://github.com/facebookresearch/MetaCLIP)|
|[EVA-CLIP](https://arxiv.org/pdf/2303.15389)|EVA-01-CLIP-g/14, EVA-01-CLIP-g/14+, EVA-02-CLIP-B/16, EVA-02-CLIP-E/14, EVA-02-CLIP-E/14+, EVA-02-CLIP-L/14, EVA-02-CLIP-L/14+|[sheet](https://docs.google.com/spreadsheets/d/1RpDAldg4K_jKuIbJWF8ocj2chLNlKMhFx_3smFowpPE/edit?usp=sharing)|[link](https://github.com/baaivision/EVA/blob/master/EVA-CLIP/README.md)|
|[EVA-CLIP-18B](https://arxiv.org/pdf/2402.04252)|EVA-CLIP-8B|[sheet](https://docs.google.com/spreadsheets/d/1RpDAldg4K_jKuIbJWF8ocj2chLNlKMhFx_3smFowpPE/edit?usp=sharing)|[link](https://github.com/baaivision/EVA/blob/master/EVA-CLIP-18B/README.md)|
|[CLIPA-v2](https://arxiv.org/pdf/2306.15658)|CLIPA(v2)-ViT-G/14, CLIPA(v2)-ViT-G/14@336px, CLIPA(v2)-ViT-H/14, CLIPA(v2)-ViT-H/14@336px (DataComp-1B), CLIPA(v2)-ViT-H/14@336px (LAION-2B), CLIPA(v2)-ViT-L/14, CLIPA(v2)-ViT-L/14@336px|[sheet](https://docs.google.com/spreadsheets/d/1VzufusSfiTUYZTH_dr8ZYtep8IPb7TnjycH0hcC438E/edit?usp=sharing)|[link](https://github.com/LAION-AI/CLIP_benchmark)|
|[$M^2$-Encoder](https://arxiv.org/pdf/2401.15896)|$M^2$-Encoder-0.4B, $M^2$-Encoder-1B, $M^2$-Encoder-10B|[sheet](https://docs.google.com/spreadsheets/d/1jF8DvZ9wsrtMbSE3V229M25uZeXFiOQeFBxe0bpFdKs/edit?usp=sharing)|[link](https://github.com/alipay/Ant-Multi-Modal-Framework/blob/main/prj/M2_Encoder/README.md)|
|[CoCa](https://arxiv.org/pdf/2205.01917v2)|CoCa-ViT-B/32, CoCa-ViT-L/14 (laion2b_s13b_b90k), CoCa-ViT-L/14(laion2b_s13b_b90k + mscoco)|[sheet](https://docs.google.com/spreadsheets/d/18z7nEjNeIlmQAf1sMBQz6Ig-E9kOxdHFx1V--MLDTUg/edit?usp=sharing)|[link](https://github.com/mlfoundations/open_clip)|
|[SigLIP](https://arxiv.org/pdf/2303.15343)|SigLIP-ViT-B/16, SigLIP-ViT-B/16@256px, SigLIP-ViT-B/16@384px, SigLIP-ViT-B/16@512px, SigLIP-ViT-L/16@256px,  SigLIP-ViT-L/16@384px, SigLIP-ViT-SO400M, SigLIP-ViT-SO400M@384px|[sheet](https://docs.google.com/spreadsheets/d/16c0Xw_Q-F0SwcdVHL5t3B0tozDkNm3iZy-78EM_EEiU/edit?usp=sharing)|[link](https://github.com/mlfoundations/open_clip)|
|[OpenCLIP](https://arxiv.org/pdf/2212.07143)|OpenCLIP-ViT-B/16, OpenCLIP-ViT-B/32@256px, OpenCLIP-ViT-L/14 (laion2b_s32b_b82k), OpenCLIP-ViT-L/14 (datacomp_xl_s13b_b90k), OpenCLIP-ViT-H/14, OpenCLIP-ViT-H/14-quickgelu, OpenCLIP-ViT-H/14-quickgelu@378px, OpenCLIP-ViT-G/14|[sheet](https://docs.google.com/spreadsheets/d/1EJxlRFmxvVR3MhiCANRJ9YI3okEleAeLdfQpkfyS824/edit?usp=sharing)|[link](https://github.com/mlfoundations/open_clip)|
|[ALIBEF](https://arxiv.org/pdf/2107.07651)|ALBEF (4M), ALBEF (14M), ALBEF (14M + coco_finetuned), ALBEF (14M + flickr_finetuned)|[sheet](https://docs.google.com/spreadsheets/d/1O89R8glW-Vp4xLcLGVyEgrfUjs-ufjgSjebSWFT9pLw/edit?usp=sharing)|[link](https://github.com/salesforce/ALBEF)|

## Overall Analysis sheet @[link](https://docs.google.com/spreadsheets/d/1j33ixMwV3DUxafYisGnBAuUzomVSfy_8JeMOpQ18Lao/edit?usp=sharing)


## WAR 
Dataset weights. 

| Dataset                        | Weight              |
|--------------------------------|---------------------|
| Imagenet                       | 0.15556157429688613 |
| ImageNet-A                     | 0.970498446080589   |
| ImageNet-V2                    | 0.2854574367981364  |
| ImageNet-R                     | 0.01                |
| ImageNet-Sketch                | 0.021456095637452655|
| Caltech101 (300 x 200)         | 0.01                |
| DTD split-1 (300x300 - 640x640)| 0.505922498560715   |
| Food101 (512*512)              | 0.01                |
| SUN397                         | 0.407563119725743   |
| Stanford Cars (360x240)        | 0.13583821249199218 |
| FGVC Aircraft                  | 0.8229545014750042  |
| Oxford Pets                    | 0.08995285864599148 |
| Flowers102                     | 0.08972060770047119 |
| EuroSAT                        | 1.0                 |
| UCF101                         | 0.01                |

## Diffusion generated synthetic Dataset

Total 7,000 captions were used to generate images. These captions were randomly sampled google caption dataset and are placed in 
https://github.com/shyammarjit/LR0.FM/tree/main/Captions

Feeding the dataset to the Diffusuion model via : 
```
import torch 
from diffusers import PixArtAlphaPipeline
pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
pipe = pipe.to('cuda')


line = line.strip() ## caption line 
offset = 0 
for fold in range(5):
    images =pipe(line, num_images_per_prompt=10,  ).images
    [img.save(f"{ROOT}/{k+1 + offset}/{i}.png") for k,img in enumerate(images)]
    offset += 10
```

## Lr-Tokens 

        
