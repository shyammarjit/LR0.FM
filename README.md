<div align="center">

## 🚀 LR0.FM (ICLR-25 🎉)<br> [webpage](https://ucf-crcv.github.io/lr0.fm/) | [paper](https://arxiv.org/abs/2502.03950) | [video](https://recorder-v3.slideslive.com/#/share?share=99927&s=b52e48b7-e501-45c7-b7c9-b1d415e77f1e) | [results]() | [weights]()<br><br> <p align="left">💡 Highlights</p>
</div>
✨ We introduce LR0.FM, a comprehensive benchmark evaluating the impact of low resolution on the zero-shot classification performance of 10 FM(s) across 66 backbones and 15 datasets.<br/>
✨ We propose a novel metric, Weighted Aggregated Robustness, to address the limitations of existing metrics and better evaluate model performance across resolutions and datasets.<br/>
✨ Our key findings show that: (i) model size positively correlates with robustness to resolution degradation, (ii) pre-training dataset quality is more important than its size, and (iii) fine-tuned and higher resolution models are less robust against LR.<br/>
✨ Our analysis further reveals that the model makes semantically reasonable predictions at LR, and the lack of fine-grained details in input adversely impacts the model’s initial layers more than the deeper layers.<br/>
✨ Our proposed LR-TK0 enhances model robustness to low-resolution without altering pre-trained weights, demonstrating effectiveness across several datasets and its generalization capability across backbones and other approaches.<br/>
</div>


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

        
