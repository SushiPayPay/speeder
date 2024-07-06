from typing import List
import os

from PIL import Image

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from torch.utils.data import Dataset

from datasets import load_dataset

INPUT_DIR = '/data/clean/cifar10'
OUTPUT_DIR = '/data/clean/cifar10-ffcv'

# IMAGE FOLDER

datasets = {
    'train': torchvision.datasets.ImageFolder(os.path.join(INPUT_DIR, 'train')),
    'test': torchvision.datasets.ImageFolder(os.path.join(INPUT_DIR, 'test'))
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for (name, ds) in datasets.items():
    writer = DatasetWriter(os.path.join(OUTPUT_DIR, f'{name}.beton'), {
        'image': RGBImageField(write_mode='jpg'),
        'label': IntField()
    }, num_workers=40)
    writer.from_indexed_dataset(ds, chunksize=400)

# PYTORCH DATASET

# datasets = {
#     'train': torchvision.datasets.CIFAR100(os.path.join(INPUT_DIR, 'train'), train=True, download=True),
#     'test': torchvision.datasets.CIFAR100(os.path.join(INPUT_DIR, 'test'), train=False, download=True)
# }

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# for (name, ds) in datasets.items():
#     writer = DatasetWriter(os.path.join(OUTPUT_DIR, f'{name}.beton'), {
#         'image': RGBImageField(write_mode='raw'), # ['raw','jpg']
#         'label': IntField()
#     }, num_workers=40)
#     writer.from_indexed_dataset(ds, chunksize=400)

# HUGGINGFACE DATASET

# datasets = load_dataset('uoft-cs/cifar10', cache_dir=INPUT_DIR)

# class HFDS(Dataset):
#     def __init__(self, ds):
#         self.ds = ds

#     def __getitem__(self, idx):
#         idx = int(idx)
#         d = self.ds[idx]
#         return d['img'].convert('RGB'), int(d['label'])
    
#     def __len__(self):
#         return len(self.ds)

# datasets = {
#     'train': HFDS(datasets['train']),
#     'test': HFDS(datasets['test'])
# }

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# for (name, ds) in datasets.items():
#     writer = DatasetWriter(os.path.join(OUTPUT_DIR, f'{name}.beton'), {
#         'image': RGBImageField(write_mode='jpg'), # ['raw','jpg']
#         'label': IntField()
#     }, num_workers=40)
#     writer.from_indexed_dataset(ds, chunksize=400)

# CINIC10

import torchvision
import torch
from torchvision import transforms
from cinic10 import CINIC10


trainset = CINIC10(root='./data', train=True, download=True,)

testset = CINIC10(root='./data', train=False, download=True)

# datasets = {
#     'train': torchvision.datasets.CIFAR100(os.path.join(INPUT_DIR, 'train'), train=True, download=True),
#     'test': torchvision.datasets.CIFAR100(os.path.join(INPUT_DIR, 'test'), train=False, download=True)
# }

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# for (name, ds) in datasets.items():
#     writer = DatasetWriter(os.path.join(OUTPUT_DIR, f'{name}.beton'), {
#         'image': RGBImageField(write_mode='raw'), # ['raw','jpg']
#         'label': IntField()
#     }, num_workers=40)
#     writer.from_indexed_dataset(ds, chunksize=400)

# EVERYTHING COMES FROM HF
# RAW VERSION SHOULD BE LOADABLE USING IMAGEFOLDER
# FFCV VERSION SHOULD BE IN BETONS

# /data/clean/<ds>/raw
# /data/clean/<ds>/ffcv

# <ds>

# mnist
# celeba
# cifar10
# cifar100
# tiny-imagenet
# imagenet
# cinic10
# stl10

# all have 'train' and 'test'
# you can create the val at load time with ffcv 'indices'

'''
class CleanDataset:
    def __init__(self, ds):
        self.ds = ds
        self.len = len(self.ds) # might not just be this

    # ADD TYPE HINTS
    def __getitem__(self, idx):
        return X,y

    def __len__(self) -> int:
        return self.len
'''

