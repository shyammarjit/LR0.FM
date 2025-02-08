# LR0.FM


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

        
