import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

    

class LR_dataset(Dataset):
    def __init__(self, root=None, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, lr_start=-1, lr_end=-1, tokenizer=None, **args):

        self.load_multiple_variants = False 
        self.load_multiple_variants = False
        
        self.load_50k = False  
        self.load_2k = False 
        self.load_7k = True  

        self.load_7k_1K = False 
        self.load_7k_3K = False 
        self.load_7k_5K = False   
        self.load_7k_7K = True 
        self.load_7k_9K = True 

        
        self.multi_scale = True 
        self.multi_scale_random = False
        self.multi_scale_16 = False  
        self.multi_scale_16_64 = False 
        self.multi_scale_16_128 = False  

        self.transform = transform
        self.image_size = image_size
        preprocess_txt = lambda text: tokenizer(text)[0]

        # self.images = [os.path.join(root, e) for e in os.listdir(root)]
        self.images = [] 
        self.labels = []
        index = []

        if self.load_7k:
            with open (f"{root}/caption_2k.txt", "r") as f:
                Lines = f.readlines()[:-1]
            with open (f"{root}/caption_5k.txt", "r") as f:
                Lines += f.readlines()
        elif self.load_2k:
            with open (f"{root}/caption_2k.txt", "r") as f:
                Lines = f.readlines()[:-1]
            
         
        captions = [e.strip() for e  in Lines] 
        self.string_captions  = [e for e in captions] 
        self.hr_captions  = [preprocess_txt(e) for e in captions] 
        self.lr_captions  = [preprocess_txt("low resolution image of " + e) for e in captions] 
        self.lr_captions1  = [preprocess_txt("Pixelated image of " + e) for e in captions] 
        self.lr_captions2  = [preprocess_txt("Downsampled image of " + e) for e in captions] 
        self.lr_captions3  = [preprocess_txt("Low Quality image of  " + e) for e in captions] 

        if self.load_7k:
            samples_2k = os.path.join(root, "Mul_samples_50_2K")
            samples_5k = os.path.join(root, "Mul_samples_50_5K")
            data_srcs = [samples_2k, samples_5k]
            flags_5K = [0, 1]
        elif self.load_2k:
            samples_10 = os.path.join(root, "Mul_samples_10_2K")
            samples_30 = os.path.join(root, "Mul_samples_30_2K")
            samples_50 = os.path.join(root, "Mul_samples_50_2K")
            data_srcs = [samples_10, samples_30, samples_50]
            
        if self.load_7k:
            for k,series in enumerate(data_srcs):
                flag_5k = flags_5K[k]
                for folder in os.listdir(series):
                    if self.load_7k_1K and int(folder) > 10:
                        continue
                    elif self.load_7k_3K and int(folder) > 30:
                        continue                 
                    folder_path = os.path.join(series, folder)
                    max_range = 2000
                    start_index = 0
                    if flag_5k:
                        max_range = 5000
                        start_index += 2000
                    for i in range(max_range):
                        img = f"rohit_caption_{i}.png"
                        img_path = os.path.join(folder_path, img)
                        assert os.path.exists(img_path)
                        self.images.append( img_path )
                        self.labels.append( start_index + int(img.split("_")[-1][:-4]) )
        elif self.load_2k:
            done = 0
            
            if self.load_7k_1K:to_be_completed = 10 
            elif self.load_7k_3K:to_be_completed = 30
            elif self.load_7k_5K:to_be_completed = 50
            elif self.load_7k_7K:to_be_completed = 70
            elif self.load_7k_9K:to_be_completed = 90

            for k,series in enumerate(data_srcs):
                if done == to_be_completed: break 
                for folder in os.listdir(series):
                    if done == to_be_completed: break 
                    done += 1
                    folder_path = os.path.join(series, folder)
                    max_range = 2000
                    start_index = 0
                    for i in range(max_range):
                        img = f"rohit_caption_{i}.png"
                        img_path = os.path.join(folder_path, img)
                        assert os.path.exists(img_path)
                        self.images.append( img_path )
                        self.labels.append( start_index + int(img.split("_")[-1][:-4]) )

        self.dataset_size = len(self.images)
        # self._resize_shapes = range(lr_start, lr_end)
        # self._resize_shapes = range(32, 48)

        if self.multi_scale or self.multi_scale_random:
            self._resize_shapes = [range(16, 32), range(32, 64), range(64, 128), range(128, 224)] 

        if self.multi_scale_16:
            self._resize_shapes = [range(16, 32)] 
        if self.multi_scale_16_64:
            self._resize_shapes = [range(16, 32), range(32, 64)] 
        if self.multi_scale_16_128:
            self._resize_shapes = [range(16, 32), range(32, 64), range(64, 128)] 

        print( f"**** {self._resize_shapes} ****")
        print( f"**** Captions : {len(self.labels)} Images : {len(self.images)} ****")

        logging.info(f"**** {self._resize_shapes} ****")
        logging.info( f"**** Captions : {len(self.labels)} Images : {len(self.images)} ****")

        idx = 0
        img, label, _ = self.__getitem__(idx)
        captions[label]
        # save_image(normalize(img), "img.png")
        

    def __getitem__(self, idx):
                    
        # print("---", idx)
        img = self.images[idx]
        label = self.labels[idx]
        img_hr = Image.open(img).convert('RGB')

        H,W = img_hr.size
        
        if self.multi_scale:
            lr_shapes = [random.choice(e) for e in self._resize_shapes]
            img_lrs = [img_hr.resize((lr_shape, lr_shape)).resize((H, W)) for lr_shape in lr_shapes]
            img_hr = self.transform(img_hr)
            img_lr = [self.transform(img_lr) for img_lr in img_lrs]
            img = torch.stack([img_hr] + img_lr)
        elif self.multi_scale_random:
            lr_shapes = random.choice( self._resize_shapes )
            lr_shapes = random.choice( lr_shapes )
            img_lr = img_hr.resize((lr_shapes, lr_shapes)).resize((H, W))

            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)

            img = torch.stack([img_hr, img_lr])
        else:
            lr_shape = random.choice(self._resize_shapes)
            img_lr = img_hr.resize((lr_shape, lr_shape)).resize((H, W))
            # img_hr.save("temp.png") , img_lr.save("temp2.png")
            img_hr = self.transform(img_hr)
            img_lr = self.transform(img_lr)
        
            img = torch.stack([img_hr, img_lr])

        return img, label
        
