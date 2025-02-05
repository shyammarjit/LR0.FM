pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

"""
This script is for CLIP 
"""
import numpy as np
import os
import torch
import argparse
from tqdm import tqdm
from pkg_resources import packaging
from data import create_dataset_zero_shot
from logger import setup_logger
from eva_clip import create_model_and_transforms, get_tokenizer
from zero_shot_classification import struct_output, zeroshot_classifier, accuracy, read_txt, get_classes_prompts, parse_args
import torch.nn.functional as F
# print("Torch version:", torch.__version__)
import pdb
import torchvision.transforms as transforms
from PIL import Image
import pickle
from torchvision.utils import save_image
import pandas as pd 
def save_pickle(data, name):
    # Store data (serialize)
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    # Load data (deserialize)
    with open(f'{name}.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data


def pairwise_l2_dist(x):
    x = F.normalize(x, p=2, dim=-1)
    x = x.unsqueeze(1) - x.unsqueeze(0)
    x = (x ** 2).sum(-1)
    return x


def reshape_transform(tensor, height=16, width=16):
    # print(tensor.shape)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def compute_flops(model, verbose=False, print_per_layer_stat=False, resolution =(3, 224, 224) ):

    if "-336" in args.backbone :
        resolution = (3, 336, 336)
    
    # python -m pip install thop fvcore
    from thop import profile
    from thop import clever_format
    from fvcore.nn import FlopCountAnalysis

    input = torch.randn(1, *resolution).cuda()
    macs, params = profile(model.visual.float(), inputs=(input, None))
    flops, params = clever_format([macs, params], "%.8f")
    print(f" GFLOP USING `thop` {macs/10 ** 9:.2f} MACs(G) '# of Params using thop': {params}M")
    
    flops = FlopCountAnalysis(model.visual.float(), input)
    print(f" \n\n***** FLOP TOTAL : {flops.total() / 10 ** 9}" )
    per_modules = flops.by_module()
    keyss = {key for key in per_modules if "block" in key and "attn" in key and per_modules[key] != 0 and "norm" not in key and 'proj' not in key}
    selected_per_module = {key: per_modules[key] for key in keyss}
    print(f"FLOP BY MODULES : {selected_per_module}" )
    print(f"FLOP BY MODULES & OPERATOR : {flops.by_module_and_operator()}" )

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")

    # from ptflops import get_model_complexity_info
    # import re
    # macs, params = get_model_complexity_info(model.visual.float(),  resolution , as_strings=True, print_per_layer_stat=print_per_layer_stat, verbose=verbose)
    # flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    # flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    # print('Computational complexity: {:<8}'.format(macs))
    # print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    # print('Number of parameters: {:<8}'.format(params))    
    quit()


def main(args):
    dumping = False 
    dump_final_feats = False    
    grad_cam = False
    flop_compute = False  
    Images_predictions = False 
    prediction_per_class = False 
    prediction_real_world = True 

    # load the model.
    if grad_cam or flop_compute or prediction_real_world:
        model, _, _ = create_model_and_transforms(args.backbone, pretrained, force_custom_clip=True, no_xattn=True, lr_clip=args.lr_mode,)    
        args.batch_size = 10 
    else:
        model, _, _ = create_model_and_transforms(args.backbone, pretrained, force_custom_clip=True, lr_clip=args.lr_mode,)    
    model = model.cuda()

    if flop_compute:
        compute_flops(model)    
    
    
    # print(args.lr_mode, args.lr_wt )
    if args.lr_mode and args.lr_wt :
        visual_checkpoint_path = args.lr_wt
        text_checkpoint_path = ''    
        checkpoint = torch.load(args.lr_wt, map_location='cpu')
        visual_incompatible_keys = model.load_state_dict(checkpoint['state_dict'], strict=args.strict)
        # visual_incompatible_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(visual_incompatible_keys)
    

    # Preparing DATASET labels and prompts
    classes, templates = get_classes_prompts(args) # print(len(classes), len(templates)) # 1000 80
    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    test = create_dataset_zero_shot(args.dataset, dtd_split=args.dtd_split, low_resolution=args.low_resolution,\
        org_resolution=336 if args.backbone=='EVA02-CLIP-L-14-336' else 224, root=args.dataset_dir, 
    )
    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Creating zero-shot classifier weights
    zeroshot_weights = zeroshot_classifier(classes, templates, model, args.backbone) # torch.Size([512, 1000])


    
    similarity = []
    labels = []
    name = f"{args.dataset}-{args.backbone}-{args.low_resolution}"
    if args.lr_mode:
        name = f"{args.dataset}-{args.backbone}-LR-TK-{args.low_resolution}"

    if dumping or dump_final_feats :
        root= 'Analysis/feat_dump'
        if dump_final_feats:name += "-Final"
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images = images.cuda() # torch.Size([400, 3, 224, 224])
                target = target.cuda() # torch.Size([400])
                # predict
                x , features = model.visual(images, intermediate_feats=True)
                if dump_final_feats:
                    sims = x
                else:
                    # for e in features:e.shape
                    sims = torch.stack(features)
                sims = F.normalize(sims, dim=-1)
                similarity .append(sims)
                labels.append(target)
        if dump_final_feats:
            similarity = torch.cat(similarity)
            print(similarity.shape)
            labels = torch.cat(labels)
            save_pickle({"feats":similarity, "labels": labels}, f'{root}/{name}')
        else:
            similarity = torch.cat(similarity, 1)
            print(similarity.shape)
            save_pickle(similarity, f'{root}/{name}')
    elif grad_cam:
        root = 'Analysis/GradCam'
        try:
            os.mkdir(root) 
        except:
            _ = 0 

        name += "-GCAM"
        from pytorch_grad_cam import GradCAM
        import cv2
        from pytorch_grad_cam.utils.image import show_cam_on_image
        model.eval()
        preprocess_temp = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])
        for param in model.visual.parameters():
            if not param.requires_grad : print(param.shape)
            param.requires_grad = True 
            
        target_layer=([model.visual.blocks[-1].norm1])
        if args.backbone == "EVA02-CLIP-B-16":
            from functools import partial
            cam = GradCAM(model=model.visual,  target_layers=target_layer, reshape_transform=partial(reshape_transform, height=14, width=14))
        else:
            cam = GradCAM(model=model.visual,  target_layers=target_layer, reshape_transform=reshape_transform)
        cam.batch_size = args.batch_size
        for i, (images, target) in enumerate(loader):     
            images = images.cuda() # torch.Size([10, 3, 224, 224])
            target = target.cuda() # torch.Size([10])
            grayscale_cam = cam(input_tensor=images, targets=None, eigen_smooth=False, aug_smooth=False)
            rgb_images = normalize(images).cpu().numpy()
            for i in range(len(rgb_images)):
                cam_image = show_cam_on_image(rgb_images[i].transpose(1,2,0), grayscale_cam[i], image_weight=0.3)
                cv2.imwrite(f'{root}/{name}_{i}.jpg', cam_image)
                save_image(normalize(images)[i], f'{root}/{args.dataset}_{classes[target[i]]}_{i}_RGB.jpg')
            break 
    elif Images_predictions:
        preds_classes = []
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.cuda() # torch.Size([400, 3, 224, 224])
                target = target.cuda() # torch.Size([400])
                
                # predict
                image_features = model.encode_image(images) # torch.Size([400, 512]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                preds = logits.topk(5)[1]
                preds = torch.cat([target.unsqueeze(-1), preds], 1)
                preds_classes.append(preds)

                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100 

        print(f"Top-1 accuracy: {top1:.2f}")
        print(f"Top-5 accuracy: {top5:.2f}")
        # Top-1 accuracy: 92.23
        # Top-5 accuracy: 99.81   
        preds_classes = torch.cat(preds_classes)
        df = pd.DataFrame( preds_classes.cpu().numpy() )
        df.to_csv(f"pred_{args.low_resolution}.csv")
    elif prediction_per_class:
        shortlisting = False 
        shortlisting = True 
        specific_img = 47729 
        try:
            os.mkdir("Analysis/Dump/") 
        except:
            _ = 0 

        pred_16 = pd.read_csv("Analysis/pred_16.csv", names=["Image_index", "true", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"], skiprows=[0]).set_index('Image_index')
        pred_32 = pd.read_csv("Analysis/pred_32.csv", names=["Image_index", "true", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"], skiprows=[0]).set_index('Image_index')
        pred_64 = pd.read_csv("Analysis/pred_64.csv", names=["Image_index", "true", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"], skiprows=[0]).set_index('Image_index')
        pred_128 = pd.read_csv("Analysis/pred_128.csv", names=["Image_index", "true", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"], skiprows=[0]).set_index('Image_index')
        pred_224 = pd.read_csv("Analysis/pred_224.csv", names=["Image_index", "true", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5"], skiprows=[0]).set_index('Image_index')
        
        True_224 = pred_224.true == pred_224.pred_1

        def return_predictions(pred):
            True_x = pred.true == pred.pred_1
            incorrect = True_224 & ~True_x
            correct = True_224 & True_x
            return incorrect, correct

        inccorect_16, correct_16 = return_predictions(pred_16)
        inccorect_32, correct_32 = return_predictions(pred_32)
        inccorect_64, correct_64 = return_predictions(pred_64)
        inccorect_128,correct_128 = return_predictions(pred_128)

        if shortlisting:
            selected = inccorect_32 & inccorect_16 & inccorect_64 & inccorect_128
            # selected = inccorect_32 & inccorect_16 
            selected = inccorect_32 & inccorect_16 
        
            images_224 = pred_224[selected].reset_index()
            images_128 = pred_128[selected].reset_index()
            images_64 = pred_64[selected].reset_index()
            images_32 = pred_32[selected].reset_index()
            images_16 = pred_16[selected].reset_index()
        else:
            images_224 = pred_224.reset_index()
            images_128 = pred_128.reset_index()
            images_64 = pred_64.reset_index()
            images_32 = pred_32.reset_index()
            images_16 = pred_16.reset_index()

        for image_index, true_label, cl_128, cl_64, cl_32, cl_16 in zip(images_224.Image_index, images_224.true, images_128.pred_1, images_64.pred_1, images_32.pred_1, images_16.pred_1  ):
            print(image_index , end="\r")
            # if image_index % 2 == 0 :
            #     continue
            # image_index, true_label, cl_128, cl_64, cl_32, cl_16
            if args.low_resolution == 224:
                selected_label = classes[true_label]
            elif args.low_resolution == 128:
                selected_label = classes[cl_128]
            elif args.low_resolution == 64:
                selected_label = classes[cl_64]
            elif args.low_resolution == 32:
                selected_label = classes[cl_32]
            elif args.low_resolution == 16:
                selected_label = classes[cl_16]  
            if specific_img is None:     
                try:
                    save_image(normalize ( test[image_index][0] ) , f"Analysis/Dump/img_{image_index}-{selected_label}-{args.low_resolution}.png")
                except:
                    continue
            else:
                if specific_img == image_index:
                    save_image(normalize ( test[image_index][0] ) , f"Analysis/Dump/img_{image_index}-{selected_label}-{args.low_resolution}.png")            
    elif prediction_real_world:
        preds_classes = []
        preds_index = []
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.cuda() # torch.Size([400, 3, 224, 224])
                
                
                # predict
                image_features = model.encode_image(images) # torch.Size([400, 512]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                preds = logits.topk(5)[1]
                preds_classes.append(preds)
                preds_index.append(target)

        dict = loader.dataset.images_index
        # ['deer_0.jpeg', 'galaxy_0.png', 'deer_1.jpeg', 'duck_0.jpg', 'gun_0.jpg', 'people_0.jpg']
        preds_classes = torch.cat(preds_classes)
        preds_index = torch.cat(preds_index)
        for i,row in enumerate(preds_classes): 
            index = preds_index[i].item()
            img = dict[index]
            row = [classes[k.item()] for k in row]
            print( f"{img} .... {row}")
            

        
        


            
  

if __name__ == "__main__":
    args = parse_args()
    main(args)