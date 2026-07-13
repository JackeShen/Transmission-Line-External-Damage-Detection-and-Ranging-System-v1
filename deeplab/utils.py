import random

import numpy as np
import torch
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        'mobilenetv3':'https://github.com/xiaolai-sqlai/mobilenetv3/blob/master/300_act3_mobilenetv3_small.pth',
         'convnext': 'https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth',
          'convnextv2': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt',
          'swintransformer':'https://objects.githubusercontent.com/github-production-release-asset-2e65be/357198522/3437414f-dd50-4288-bf41-b7a87a940980?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241121%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241121T090100Z&X-Amz-Expires=300&X-Amz-Signature=d466e93376df56bb33bfa370b3be76bad415788f9170f86a4d3646fef934b888&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dswinv2_tiny_patch4_window8_256.pth&response-content-type=application%2Foctet-stream',
          'efficientnet': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
        'mobilemetmanba': 'https://drive.usercontent.google.com/download?id=1KevdesGm8Pb4fEbgZ5JY4FMy9IQgYbiZ&export=download&confirm=t&uuid=0891b3f5-deb9-4e9d-9e9d-33cf6e1f4b7c',}
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)