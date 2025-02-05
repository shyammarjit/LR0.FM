# https://colab.research.google.com/gist/JingyunLiang/a5e3e54bc9ef8d7bf594f6fee8208533/swinir-demo-on-real-world-image-sr.ipynb#scrollTo=IRDbDYYMQt_Y

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

from logger import setup_logger
from eva_clip import create_model_and_transforms, get_tokenizer
from data import create_dataset_zero_shot, _convert_image_to_rgb
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# print("Torch version:", torch.__version__)
from torchvision.utils import save_image
from PIL import Image
import sys 
home_directory = os.path.expanduser('~')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from zero_shot_classification import struct_output, get_classes_prompts, zeroshot_classifier,\
    accuracy, read_txt, get_classes_prompts

def normalize(x):return (x - x.min()) / (x.max() - x.min())

def create_bsrgan():
    root = f"{home_directory}/resolution-bm/SR_MODELS/BSRGAN"
    sys.path.append( root )
    from models.network_rrdbnet import RRDBNet as net

    model_name = "BSRGAN"
    sf = 4
    model_path = os.path.join(root, 'model_zoo', model_name+'.pth')          # set model path
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()
    produce = lambda x: model(x)
    return model, produce 

def create_realESRGAN():
    root = f"{home_directory}/resolution-bm/SR_MODELS/Real-ESRGAN"
    root_wt = f"{home_directory}/resolution-bm/SR_MODELS/"
    sys.path.append( root )
    
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.download_util import load_file_from_url
    from realesrgan import RealESRGANer
    model_name = 'RealESRGAN_x4plus'
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    outscale = 4
    tile = 800
    tile_pad = 10 
    pre_pad = 0 
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    model_path = os.path.join(root_wt, 'experiments', 'pretrained_models', 'RealESRGAN_x4plus.pth')
    # model_path = os.path.join('weights', model_name + '.pth')
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    dni_weight = None
    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=True,
        gpu_id=None)

    # x = images
    # x = (x * 255).cpu().numpy().astype(np.uint8)
    # Image.fromarray(x.transpose(0,2,3,1)[0].astype('uint8'), 'RGB').save("temp.png")
    # Image.fromarray( x.transpose(0,2,3,1) ).convert('RGB')

    sr_produce = lambda x : upsampler.enhance( x  , outscale=outscale)
    torch.cuda.empty_cache()
    return model, sr_produce
    
def create_SwinIR():
    parent = f"{home_directory}/resolution-bm/SR_MODELS/"
    root = f"{home_directory}/resolution-bm/SR_MODELS/SwinIR"
    sys.path.append( root )
    
    large_model = True 
    model_path = "experiments/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
    model_path = os.path.join(parent, model_path)
    
    # large_model = False
    # model_path = "experiments/pretrained_models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    # model_path = os.path.join(parent, model_path)
    
    from models.network_swinir import SwinIR as net
    from main_test_swinir import define_model
    from main_test_swinir import test as test_sr
    
    task = 'real_sr'
    scale = 4
    
    training_patch_size = 128
    tile = 640
    border = 0
    window_size = 8
    tile_overlap = 32
    model = define_model(task, scale, large_model, model_path, training_patch_size=training_patch_size)
    model.eval()
    model = model.to(device)
    for k, v in model.named_parameters():
        v.requires_grad = False
    
    def preprocessor(img_lq):
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = test_sr(img_lq, model, tile, window_size, tile_overlap, scale)
        output = output[..., :h_old * scale, :w_old * scale]
        return output

    # img = cv2.imread('/home/priyank/resolution-bm/EVA/EVA-CLIP/rei/temp3.png', cv2.IMREAD_UNCHANGED)
    sr_produce = lambda x : preprocessor(x)
    torch.cuda.empty_cache()
    return model, sr_produce

def create_Inf_DiT():
    parent = f"{home_directory}/resolution-bm/SR_MODELS/"
    root = f"{home_directory}/resolution-bm/SR_MODELS/Inf-DiT"
    sys.path.append( root )

    from dit.model import DiffusionEngine
    from sat.model.base_model import get_model
    config_file = f'{root}/configs/text2image-sr.yaml'
    config_dict = dict(attention_dropout=0.0, augment_dim=0, batch_from_same_dataset=False, batch_size=4, bf16=True, block_batch=4, block_size=10000, checkpoint_activations=False, checkpoint_num_layers=1, checkpoint_skip_layers=0, config_path=config_file, 
        crop_size=0, cross_adaln=False, cross_attn_hidden_size=640, cross_lr=True, cuda=True, ddpm_time_emb=False, deepscale=False, deepscale_config=None, deepspeed=False, deepspeed_mpi=False, device=0, distributed_backend='nccl', drop_path=0.0, epochs=None, 
        eval_batch_size=None, eval_interval=None, eval_iters=100, exit_interval=None, experiment_name='generate', fp16=False, gradient_accumulation_steps=1, hidden_dropout=0.0, hidden_size=1280, hidden_size_per_attention_head=None, image_block_size=128, image_condition=True, 
        image_size=512, in_channels=6, infer_sr_scale=4, inference_batch_size=1, inference_type='full', init_noise=True, inner_hidden_size=None, input_path='input.txt', input_source='interactive', input_time='adaln', input_type='cli', is_decoder=False, is_gated_mlp=False, 
        iterable_dataset=False, label_dim=0, label_dropout=0, layernorm_epsilon=1e-06, layernorm_order='pre', length_penalty=0.0, load=None, local_rank=0, log_interval=50, lr=0.0001, lr_decay_iters=None, lr_decay_ratio=0.1, lr_decay_style='linear', lr_dropout=0, lr_size=0, 
        make_vocab_size_divisible_by=128, master_ip='localhost', master_port='34923', max_inference_batch_size=12, max_sequence_length=256, min_tgt_length=0, mode='inference', model_parallel_size=1, network='ckpt/mp_rank_00_model_states.pt', no_concat=False, no_crossmask=True, 
        no_load_rng=False, no_repeat_ngram_size=0, no_save_rng=False, nogate=True, num_attention_heads=16, num_beams=1, num_layers=28, num_multi_query_heads=0, num_steps=18, num_workers=1, out_channels=3, out_dir='samples', out_seq_length=256, output_path='./samples', 
        patch_size=4, prefetch_factor=4, profiling=-1, qk_ln=True, random_direction=False, random_position=True, rank=0, re_position=True, reg_token_num=0, resume_dataloader=False, round=32, save=None, save_args=False, save_interval=5000, scale_factor=1, seed=15987, skip_init=False, 
        split='1000,1,1', sr_scale=4, stop_grad_patch_embed=False, strict_eval=False, summary_dir='', temperature=1.0, test_data=None, text_dropout=0, tokenizer_type="'fake'", top_k=0, top_p=0.0, train_data=None, train_data_weights=None, train_iters=10000, use_gpu_initialization=False, 
        valid_data=None, vector_dim=768, vocab_size=1, warmup=0.01, weight_decay=0.01, with_id=False, world_size=1, zero_stage=0)
    namespace = argparse.Namespace(**config_dict)
    net = get_model(namespace, DiffusionEngine).to(device)

    network_wt = f"{parent}/experiments/pretrained_models/mp_rank_00_model_states.pt"
    data = torch.load(network_wt, map_location='cpu')
    net.load_state_dict(data['module'], strict=False)
    print('Loading Fished!')

    net.eval()
    net = net.to(device)
    for k, v in net.named_parameters():
        v.requires_grad = False
    inference_type = torch.bfloat16
    ar = namespace.inference_type == 'ar'
    ar2 = namespace.inference_type == 'ar2'
        
    resize_fn =  transforms.Resize(32, interpolation=InterpolationMode.BICUBIC)
    def preprocessor(img_lq):
        img_lq = resize_fn(img_lq)
        img_lq = img_lq  * 2 - 1

        H, W = img_lq.shape[-2:]
        new_h = H *namespace.infer_sr_scale
        new_w = W *namespace.infer_sr_scale

        tmp_lr_image = transforms.functional.resize(img_lq, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
        concat_lr_image = torch.clip(tmp_lr_image, -1, 1).to(device).to(inference_type)
        lr_image = img_lq.to(inference_type)
        
        collect_attention = False
        samples = net.sample(shape=concat_lr_image.shape, images=concat_lr_image, lr_imgs=lr_image, dtype=concat_lr_image.dtype, device=device, init_noise=namespace.init_noise, do_concat=not namespace.no_concat)
        return samples.float()

    sr_produce = lambda x : preprocessor(x)
    torch.cuda.empty_cache()
    return net, sr_produce
    
def create_ADDSR():
    parent = f"{home_directory}/resolution-bm/SR_MODELS/"
    root = f"{home_directory}/resolution-bm/SR_MODELS/AddSR"
    sys.path.append( root )

    from test_addsr import load_addsr_pipeline, load_tag_model
    from pipelines.pipeline_addsr import StableDiffusionControlNetPipeline 
    from accelerate import Accelerator
    from ram import inference_ram as inference
    from utils_addr.wavelet_color_fix import wavelet_color_fix, adain_color_fix
    from utils_addr.wavelet_color_fix import adaptive_instance_normalization
    mixed_precision = "fp16"
    prompt = ""
    start_point = "lr"
    num_inference_steps = 4
    PSR_weight = 0.5
    enable_xformers_memory_efficient_attention = True 

    addsr_model_path =  os.path.join(parent, "experiments", "pretrained_models", 'addsr')
    ram_ft_path =  os.path.join(parent, "experiments", "pretrained_models", 'DAPE.pth')
    pretrained_model_path = f"{home_directory}/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-base"
    pretrained_model_path = os.path.join(pretrained_model_path, 'snapshots')
    folder_hash = os.listdir(pretrained_model_path)[0]
    pretrained_model_path = os.path.join(pretrained_model_path, folder_hash)
    
    config_dict = dict(PSR_weight=0.5, 
        added_prompt='clean, high-resolution, 8k, extremely detailed, best quality, sharp', align_method='adain', blending_alpha=1.0, conditioning_scale=1.0, guidance_scale=0, seed=None, start_point='lr', start_steps=999, upscale=4, vae_tiled_size=224,
        latent_tiled_overlap=4, latent_tiled_size=320, mixed_precision='fp16', negative_prompt='dotted, noise, blur, lowres, smooth', num_inference_steps=4, process_size=512, prompt='', sample_times=1, save_prompts=False, image_path='',  output_dir='', 
        pretrained_model_path=pretrained_model_path,  ram_ft_path=ram_ft_path, addsr_model_path=addsr_model_path, 
        )
    namespace = argparse.Namespace(**config_dict)

    accelerator = Accelerator( mixed_precision=mixed_precision)
    pipeline = load_addsr_pipeline(namespace, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(namespace, accelerator.device)

    model.eval()
    model = model.to(device)
    for k, v in model.named_parameters():
        v.requires_grad = False
    

    ram_transforms = transforms.Compose([ 
        transforms.Resize((384, 384)), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    rscale = namespace.upscale
    ori_width, ori_height = args.low_resolution, args.low_resolution
    if ori_width < namespace.process_size//rscale or ori_height < namespace.process_size//rscale:
        scale = (namespace.process_size//rscale)/min(ori_width, ori_height)
        new_h = int(scale*ori_height)
        new_w = int(scale*ori_width)
    else:
        new_h = int(ori_height)
        new_w = int(ori_width)

    new_h = int(new_h * rscale)
    new_w = int(new_w * rscale)

    generator = torch.Generator(device=accelerator.device)
    assert namespace.sample_times == 1
    
    def preprocessor(img_lq):
        validation_prompt = ""
        B = img_lq.shape[0]
        lq = ram_transforms(img_lq)
        prmpts = [] 
        ram_encoder_hidden_states = model.generate_image_embeds(lq)
        for i in range(B):
            res = inference(lq[i].unsqueeze(0), model)
            validation_prompt = f"{res[0]}, {namespace.prompt}," + namespace.added_prompt
            prmpts.append(validation_prompt)

        # print(f'{prmpts}')
        img_lq = transforms.functional.resize(img_lq, [new_h, new_w], interpolation=InterpolationMode.BICUBIC)
        
        with torch.autocast("cuda"):
            image = pipeline(
                    prmpts, img_lq, num_inference_steps=namespace.num_inference_steps, generator=generator, height=new_h, width=new_w,
                    guidance_scale=namespace.guidance_scale, negative_prompt=namespace.negative_prompt, conditioning_scale=namespace.conditioning_scale,
                    start_point=namespace.start_point, ram_encoder_hidden_states=ram_encoder_hidden_states, args=namespace, output_type="pt", 
                ).images
                # [0]

        image = adaptive_instance_normalization(image, img_lq)
        return image

    sr_produce = lambda x : preprocessor(x)
    torch.cuda.empty_cache()
    return model, sr_produce
    
def create_IDM():
    parent = f"{home_directory}/resolution-bm/SR_MODELS/"
    root = f"{home_directory}/resolution-bm/SR_MODELS/IDM"
    sys.path.append( root )

    import core.logger as Logger
    import model as Model
    import data_idm.util as Util

    model_path = "experiments/pretrained_models/cat"
    config = os.path.join(root, "config/cat_liifsr3_x16.json")
    cell_decode = None 

    # model_path = "experiments/pretrained_models/df2k"
    # config = os.path.join(root, "config/df2k_liifsr3_x4.json")
    # cell_decode = True 

    model_path = os.path.join(parent, model_path)
    config_dict = dict(
        config=config, phase="val", 
        resume=model_path, gpu_ids=0, enable_wandb=False, use_ddim=False, debug=False)

    
    namespace = argparse.Namespace(**config_dict)
    opt = Logger.parse(namespace)
    opt = Logger.dict_to_nonedict(opt)
    
    print(opt)
    # model
    if cell_decode :
        opt['model']['diffusion']['cell_decode'] = True 
    diffusion = Model.create_model(opt)
    print('Initial Model Finished')
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    diffusion.netG.eval()
    diffusion.netG = diffusion.netG.to(device)
    for k, v in diffusion.netG.named_parameters():
        v.requires_grad = False
    
    
    
    def preprocessor(img_lq):
        
        B = img_lq.shape[0]
        # B = 1
        dummy_torch = torch.zeros([B, 3, 16 * 8 , 16 * 8 ]).cuda()

        # save_image(normalize(img_lq), "temp.png"), save_image(img_lq, "temp2.png")
        val_data = {'inp': img_lq, 'gt': dummy_torch}
        # print(img_lq.shape, dummy_torch.shape)
        diffusion.feed_data(val_data)
        diffusion.test(crop=False, continous=True, use_ddim=opt['use_ddim'])

        visuals = diffusion.get_current_visuals(sample=True)
        # print(visuals["SAM"].shape)
        
        SR = visuals["SAM"][-B:].cuda()
        # save_image(normalize(SR), "temp3.png"), save_image(SR, "temp4.png")
        return normalize(SR)

    sr_produce = lambda x : preprocessor(x)
    torch.cuda.empty_cache()
    return diffusion, sr_produce

def create_HAT():
    parent = f"{home_directory}/resolution-bm/SR_MODELS/"
    root = f"{home_directory}/resolution-bm/SR_MODELS/HAT"
    sys.path.append( root )


    config = os.path.join(root, 'options/test/', 'HAT-S_SRx4.yml')
    config = os.path.join(root, 'options/test/', 'HAT-L_SRx4_ImageNet-pretrain.yml')
    config = os.path.join(root, 'options/test/', 'HAT_SRx4.yml')
    config = os.path.join(root, 'options/test/', 'HAT_SRx4_ImageNet-LR.yml')
    config = os.path.join(root, 'options/test/', 'HAT_GAN_Real_SRx4.yml')
    config = os.path.join(root, 'options/test/', 'HAT_SRx4_ImageNet-pretrain.yml')
    


    import yaml
    from collections import OrderedDict

    with open(config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict = OrderedDict(config_dict)
        print(config_dict)

         
    

    import pdb
    pdb.set_trace()
        
    # OrderedDict([('name', 'HAT_SRx4_ImageNet-pretrain'), ('model_type', 'HATModel'), ('scale', 4), ('num_gpu', 1), ('manual_seed', 0), ('datasets', OrderedDict([('test_1', OrderedDict([('name', 'Set5'), ('type', 'PairedImageDataset'), ('dataroot_gt', './datasets/Set5/GTmod4'), ('dataroot_lq', './datasets/Set5/LRbicx4'), ('io_backend', OrderedDict([('type', 'disk')])), ('phase', 'test'), ('scale', 4)]))])), ('network_g', OrderedDict([('type', 'HAT'), ('upscale', 4), ('in_chans', 3), ('img_size', 64), ('window_size', 16), ('compress_ratio', 3), ('squeeze_factor', 30), ('conv_scale', 0.01), ('overlap_ratio', 0.5), ('img_range', 1.0), ('depths', [6, 6, 6, 6, 6, 6]), ('embed_dim', 180), ('num_heads', [6, 6, 6, 6, 6, 6]), ('mlp_ratio', 2), ('upsampler', 'pixelshuffle'), ('resi_connection', '1conv')])), ('path', OrderedDict([('pretrain_network_g', './experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth'), ('strict_load_g', True), ('param_key_g', 'params_ema'), ('results_root', '/home/priyank/resolution-bm/SR_MODELS/HAT/results/HAT_SRx4_ImageNet-pretrain'), ('log', '/home/priyank/resolution-bm/SR_MODELS/HAT/results/HAT_SRx4_ImageNet-pretrain'), ('visualization', '/home/priyank/resolution-bm/SR_MODELS/HAT/results/HAT_SRx4_ImageNet-pretrain/visualization')])), ('val', OrderedDict([('save_img', True), ('suffix', None), ('metrics', OrderedDict([('psnr', OrderedDict([('type', 'calculate_psnr'), ('crop_border', 4), ('test_y_channel', True)])), ('ssim', OrderedDict([('type', 'calculate_ssim'), ('crop_border', 4), ('test_y_channel', True)]))]))])), ('dist', False), ('rank', 0), ('world_size', 1), ('auto_resume', False), ('is_train', False)])
    # OrderedDict([('name', 'HAT_SRx4_ImageNet-pretrain'), ('model_type', 'HATModel'), ('scale', 4), ('num_gpu', 1), ('manual_seed', 0), ('datasets', {'test_1': {'name': 'Set5', 'type': 'PairedImageDataset', 'dataroot_gt': './datasets/Set5/GTmod4', 'dataroot_lq': './datasets/Set5/LRbicx4', 'io_backend': {'type': 'disk'}}}), ('network_g', {'type': 'HAT', 'upscale': 4, 'in_chans': 3, 'img_size': 64, 'window_size': 16, 'compress_ratio': 3, 'squeeze_factor': 30, 'conv_scale': 0.01, 'overlap_ratio': 0.5, 'img_range': 1.0, 'depths': [6, 6, 6, 6, 6, 6], 'embed_dim': 180, 'num_heads': [6, 6, 6, 6, 6, 6], 'mlp_ratio': 2, 'upsampler': 'pixelshuffle', 'resi_connection': '1conv'}), ('path', {'pretrain_network_g': './experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth', 'strict_load_g': True, 'param_key_g': 'params_ema'}), ('val', {'save_img': True, 'suffix': None, 'metrics': {'psnr': {'type': 'calculate_psnr', 'crop_border': 4, 'test_y_channel': True}, 'ssim': {'type': 'calculate_ssim', 'crop_border': 4, 'test_y_channel': True}}})])

    import core.logger as Logger
    import model as Model
    import data_idm.util as Util

    model_path = "experiments/pretrained_models/cat"
    config = os.path.join(root, "config/cat_liifsr3_x16.json")
    cell_decode = None 

    # model_path = "experiments/pretrained_models/df2k"
    # config = os.path.join(root, "config/df2k_liifsr3_x4.json")
    # cell_decode = True 

    model_path = os.path.join(parent, model_path)
    config_dict = dict(
        config=config, phase="val", 
        resume=model_path, gpu_ids=0, enable_wandb=False, use_ddim=False, debug=False)


    namespace = argparse.Namespace(**config_dict)
    opt = Logger.parse(namespace)
    opt = Logger.dict_to_nonedict(opt)
    
    print(opt)
    # model
    if cell_decode :
        opt['model']['diffusion']['cell_decode'] = True 
    diffusion = Model.create_model(opt)
    print('Initial Model Finished')
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    diffusion.netG.eval()
    diffusion.netG = diffusion.netG.to(device)
    for k, v in diffusion.netG.named_parameters():
        v.requires_grad = False
    

    
    
    def preprocessor(img_lq):
        
        B = img_lq.shape[0]
        # B = 1
        
        dummy_torch = torch.zeros([B, 3, 16 * 8 , 16 * 8 ]).cuda()

        # save_image(normalize(img_lq), "temp.png"), save_image(img_lq, "temp2.png")
        val_data = {'inp': img_lq, 'gt': dummy_torch}
        # print(img_lq.shape, dummy_torch.shape)
        diffusion.feed_data(val_data)
        diffusion.test(crop=False, continous=True, use_ddim=opt['use_ddim'])

        visuals = diffusion.get_current_visuals(sample=True)
        # print(visuals["SAM"].shape)
        
        SR = visuals["SAM"][-B:].cuda()
        # save_image(normalize(SR), "temp3.png"), save_image(SR, "temp4.png")
        return normalize(SR)

    sr_produce = lambda x : preprocessor(x)
    torch.cuda.empty_cache()
    return diffusion, sr_produce

    

def main(args):
    # https://github.com/sunny2109/SAFMN?tab=readme-ov-file
    # https://github.com/assafshocher/ZSSR
    # https://github.com/Tencent/Real-SR
    # https://github.com/bahjat-kawar/ddrm
    # https://github.com/RisingEntropy/LPFInISR
    # https://github.com/qyp2000/XPSR
    
    
    
    
    # https://github.com/cszn/BSRGAN
    if args.SR_Model == "BSRGAN":SR_MODEL, sr_produce = create_bsrgan()
    # https://github.com/xinntao/Real-ESRGAN
    if args.SR_Model == "ESRGAN":SR_MODEL, sr_produce = create_realESRGAN()
    # https://github.com/jingyunliang/swinir?tab=readme-ov-file
    if args.SR_Model == "SwinIR":SR_MODEL, sr_produce = create_SwinIR()
    # https://github.com/NJU-PCALab/AddSR
    if args.SR_Model == "ADDSR":SR_MODEL, sr_produce = create_ADDSR()
    # https://github.com/THUDM/Inf-DiT 
    if args.SR_Model == "Inf_DiT":SR_MODEL, sr_produce = create_Inf_DiT()
    # https://github.com/Ree1s/IDM
    if args.SR_Model == "IDM":SR_MODEL, sr_produce = create_IDM()
    # https://github.com/XPixelGroup/HAT?tab=readme-ov-file
    if args.SR_Model == "HAT":SR_MODEL, sr_produce = create_HAT()
    
    


    # load the model.)    
    model, _, _ = create_model_and_transforms(args.backbone, pretrained, force_custom_clip=True, lr_clip=args.lr_mode, no_xattn=True)    
    model = model.cuda()
    
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
    # print(f" Input image resolution {args.image_resolution}, Model resolution: {input_resolution}")
    print(f" Classes: {len(classes)}, prompt templates: {len(templates)}")

    # Prepare the dataloader
    # sr_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))      
    # sr_resize = transforms.Resize(336 if args.backbone=='EVA02-CLIP-L-14-336' else 224, interpolation=InterpolationMode.BICUBIC)
    
    sr_transform_test = transforms.Compose([
                transforms.Resize(224,interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(size=(224, 224)),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

    test = create_dataset_zero_shot(args.dataset, dtd_split=args.dtd_split, low_resolution=args.low_resolution,\
        org_resolution=336 if args.backbone=='EVA02-CLIP-L-14-336' else 224, root=args.dataset_dir, enable_resize=False,
    )
    
    # len(test), len(test.images)
    # test.images = test.images[:25]
    # test.labels = test.labels[:25]
    # len(test), len(test.images)

    loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, num_workers=args.num_workers)


    # Creating zero-shot classifier weights
    zeroshot_weights = zeroshot_classifier(classes, templates, model, args.backbone) # torch.Size([512, 1000])

    
    if args.Dump:
        root = f"Analysis/SR_Images/{args.dataset}/"
        try:
            os.mkdir(root) 
        except:
            _ = 0 
        dum_raw= True
        for i, (images, target) in enumerate(tqdm(loader)):
            batch = images.shape[0]
            images = images.cuda() # torch.Size([400, 3, 224, 224])
            target = target.cuda() # torch.Size([400])            

            sr_images = sr_produce(normalize(images))

            if dum_raw:
                for i in range(batch):save_image(normalize(images)[i], f"{root}/raw_{i}_{args.low_resolution}.png")
            for i in range(batch):
                save_image(normalize(sr_images)[i], f"{root}/{args.SR_Model}_{i}_{args.low_resolution}.png")
            quit()



    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda() # torch.Size([400, 3, 224, 224])
            target = target.cuda() # torch.Size([400])
            
            if args.vanilla:
                # save_image(images, "temp2.png")
                images_stack = []
                for img in images:
                    img = transforms.functional.to_pil_image(normalize(img), mode=None)
                    img = sr_transform_test(img)
                    images_stack.append(img)
                images = torch.stack(images_stack,0).cuda()
                # save_image(images, "temp2.png")
            else:
                sr_images = sr_produce(normalize(images))
                images_stack = []
                for img in sr_images:
                    img = transforms.functional.to_pil_image(normalize(img), mode=None)
                    img = sr_transform_test(img)
                    images_stack.append(img)
                images = torch.stack(images_stack,0).cuda()
                # save_image(images, "temp3.png")

            # predict
            image_features = model.encode_image(images) # torch.Size([400, 512]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")
  
  
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--dataset",
        type=str,
        default="imagenet1k",
        help="Dataset name (small leter recombedded)",
        choices=['imagenet1k', 'imagenet_a', 'imagenet_r', 'imagenet_sketch', 'imagenet_v2', 'caltech101', 'dtd', 'food101', 'fgvc_aircraft',\
            'sun397', 'pets', 'cars', 'flowers', 'eurosat', 'ucf101', 'birdsnap'],
    )
    parser.add_argument("--batch_size",
        type=int,
        default=400,
        help="dataloader batch size",
    )
    parser.add_argument("--backbone",
        type=str,
        default='2CLIPL14-336',
        help="EVA-CLIP backbone model",
        choices=['2CLIPL14-336', '2CLIPL14', '2CLIPbigE14p', '2CLIPbigE14', '2CLIPB16', '1CLIPg14p', '1CLIPg14'],
    )
    parser.add_argument("--num_workers",
        type=int,
        default=2,
        help="num of CPU workers in dataloader",
    )              
    parser.add_argument("--low_resolution",
        type=int,
        default=224,
        help="input image resolution for model",
    )
    parser.add_argument("--output_dir",
        type=str,
        default="./",
        help="input image resolution for model",
    )
    parser.add_argument("--dtd_split",
        type=int,
        default=1,
        help="Split number for DTD dataset, for other dataset you can ignore this.",
    )
    parser.add_argument("--class_dir",
        type=str,
        default="./CLIP/dataloaders/classes/",
        help="input image resolution for model",
    )
    parser.add_argument("--templates_dir",
        type=str,
        default="./CLIP/dataloaders/templates",
        help="input image resolution for model",
    )
    
    parser.add_argument("--SR-Model", type=str, default=None, help="input image resolution for model",
    choices=["BSRGAN", "ESRGAN", "SwinIR", "Inf_DiT", "ADDSR", "IDM", "HAT"])
    

    parser.add_argument('--lr-mode', action='store_true', default=False)
    parser.add_argument('--lr-wt', type=str, default=False)
    parser.add_argument('--strict', action='store_true', default=False)
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument('--Dump', action='store_true', default=False)
    parser.add_argument('--vanilla', action='store_true', default=False)
    

    parser.add_argument("--dataset_dir",
        type=str,
        default=None,
        help="input image resolution for model",
    )
    if input_args is not None: args = parser.parse_args(input_args)
    else: args = parser.parse_args()
    
    if args.backbone == "2CLIPL14-336":
        args.backbone = 'EVA02-CLIP-L-14-336'
    elif args.backbone == '2CLIPL14':
        args.backbone = 'EVA02-CLIP-L-14'
    elif args.backbone == '2CLIPbigE14p':
        args.backbone = 'EVA02-CLIP-bigE-14-plus'
    elif args.backbone == '2CLIPbigE14':
        args.backbone = 'EVA02-CLIP-bigE-14'
    elif args.backbone == '2CLIPB16':
        args.backbone = 'EVA02-CLIP-B-16'
    elif args.backbone == '1CLIPg14p':
        args.backbone = 'EVA01-CLIP-g-14-plus'
    elif args.backbone == '1CLIPg14':
        args.backbone = 'EVA01-CLIP-g-14'
    else:
        raise ValueError(f'Wrong backbone type: {args.backbone}')
    # structure the output dir
    struct_output(args)
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)