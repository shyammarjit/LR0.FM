<div align="center">

## 🚀 LR0.FM (ICLR-25 🎉)<br> [webpage](https://ucf-crcv.github.io/lr0.fm/) | [paper](https://arxiv.org/abs/2502.03950) | [video](https://recorder-v3.slideslive.com/#/share?share=99927&s=b52e48b7-e501-45c7-b7c9-b1d415e77f1e) | [results]() | [weights]()<br><br> <p align="left">💡 Highlights</p>
</div>
✨ We introduce LR0.FM, a comprehensive benchmark evaluating the impact of low resolution on the zero-shot classification performance of 10 FM(s) across 66 backbones and 15 datasets.<br/>
✨ We propose a novel metric, Weighted Aggregated Robustness, to address the limitations of existing metrics and better evaluate model performance across resolutions and datasets.<br/>
✨ Our key findings show that: (i) model size positively correlates with robustness to resolution degradation, (ii) pre-training dataset quality is more important than its size, and (iii) fine-tuned and higher resolution models are less robust against LR.<br/>
✨ Our analysis further reveals that the model makes semantically reasonable predictions at LR, and the lack of fine-grained details in input adversely impacts the model’s initial layers more than the deeper layers.<br/>
✨ Our proposed LR-TK0 enhances model robustness to low-resolution without altering pre-trained weights, demonstrating effectiveness across several datasets and its generalization capability across backbones and other approaches.<br/>
</div>


## 🎖️ Zero-Shot Classification Results in Low Resolution

Model | Backbones | Code | Results|
| --- | --- | --- | --- |
|[CLIP](https://arxiv.org/pdf/2103.00020)|CLIP-ViT-B/32, CLIP-ViT-B/16, CLIP-ViT-L/14, CLIP-ViT-L/14@336px, CLIP-RN50, CLIP-RN101, CLIP-RN50x4, CLIP-RN50x16, CLIP-RN50x64|[code]()|[results]()|
|[BLIP](https://arxiv.org/pdf/2201.12086)|BLIP-ViT-B/16 (14M), BLIP-ViT-B/16 (129M), BLIP-ViT-B/16 & CapFilt-L (129M), BLIP-ViT-L/16 (129M), BLIP-ViT-B/16 (129M + COCO), BLIP-ViT-B/16 (129M + Flickr), BLIP-ViT-L/16 (129M + COCO), BLIP-ViT-L/16 (129M + Flickr)|[code]()|[results]()|
|[MetaCLIP](https://arxiv.org/pdf/2309.16671)|MetaCLIP-ViT-B/32 (400M), MetaCLIP-ViT-B/32 (2.5B), MetaCLIP-ViT-B/16 (400M), MetaCLIP-ViT-B/16 (2.5B), MetaCLIP-ViT-L/14 (400M), MetaCLIP-ViT-L/14 (2.5B), MetaCLIP-ViT-H/14 (2.5B), MetaCLIP-ViT-G/14 (2.5B)|[code]()|[results]()|
|[EVA-CLIP](https://arxiv.org/pdf/2303.15389)|EVA-01-CLIP-g/14, EVA-01-CLIP-g/14+, EVA-02-CLIP-B/16, EVA-02-CLIP-E/14, EVA-02-CLIP-E/14+, EVA-02-CLIP-L/14, EVA-02-CLIP-L/14+|[code]()|[results]()|
|[EVA-CLIP-18B](https://arxiv.org/pdf/2402.04252)|EVA-CLIP-8B|[code]()|[results]()|
|[CLIPA-v2](https://arxiv.org/pdf/2306.15658)|CLIPA(v2)-ViT-G/14, CLIPA(v2)-ViT-G/14@336px, CLIPA(v2)-ViT-H/14, CLIPA(v2)-ViT-H/14@336px (DataComp-1B), CLIPA(v2)-ViT-H/14@336px (LAION-2B), CLIPA(v2)-ViT-L/14, CLIPA(v2)-ViT-L/14@336px|[code]()|[results]()|
|[$M^2$-Encoder](https://arxiv.org/pdf/2401.15896)|$M^2$-Encoder-0.4B, $M^2$-Encoder-1B, $M^2$-Encoder-10B|[code]()|[results]()|
|[CoCa](https://arxiv.org/pdf/2205.01917v2)|CoCa-ViT-B/32, CoCa-ViT-L/14 (laion2b_s13b_b90k), CoCa-ViT-L/14(laion2b_s13b_b90k + mscoco)|[code]()|[results]()|
|[SigLIP](https://arxiv.org/pdf/2303.15343)|SigLIP-ViT-B/16, SigLIP-ViT-B/16@256px, SigLIP-ViT-B/16@384px, SigLIP-ViT-B/16@512px, SigLIP-ViT-L/16@256px,  SigLIP-ViT-L/16@384px, SigLIP-ViT-SO400M, SigLIP-ViT-SO400M@384px|[code]()|[results]()|
|[OpenCLIP](https://arxiv.org/pdf/2212.07143)|OpenCLIP-ViT-B/16, OpenCLIP-ViT-B/32@256px, OpenCLIP-ViT-L/14 (laion2b_s32b_b82k), OpenCLIP-ViT-L/14 (datacomp_xl_s13b_b90k), OpenCLIP-ViT-H/14, OpenCLIP-ViT-H/14-quickgelu, OpenCLIP-ViT-H/14-quickgelu@378px, OpenCLIP-ViT-G/14|[code]()|[results]()|
|[ALIBEF](https://arxiv.org/pdf/2107.07651)|ALBEF (4M), ALBEF (14M), ALBEF (14M + coco_finetuned), ALBEF (14M + flickr_finetuned)|[code]()|[results]()|

<!-- ## Overall Analysis sheet @[link](https://docs.google.com/spreadsheets/d/1j33ixMwV3DUxafYisGnBAuUzomVSfy_8JeMOpQ18Lao/edit?usp=sharing) -->


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

        



## ✏️ Citation
If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```bash
@inproceedings{pathak2025lrfm,
    title={{LR}0.{FM}: {LOW}-{RESOLUTION} {ZERO}-{SHOT} {CLASSIFICATION} {BENCHMARK} {FOR} {FOUNDATION} {MODELS}},
    author={Priyank Pathak and Shyam Marjit and Shruti Vyas and Yogesh S Rawat},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=AsFxRSLtqR}
}
```