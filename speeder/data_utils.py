import os

import torch
from torchvision import transforms, datasets

from ffcv.fields.decoders import IntDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.fields.rgb_image import \
    CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.transforms import \
    RandomHorizontalFlip, \
    ToDevice, \
    ToTensor, \
    ToTorchImage, \
    Squeeze, \
    Convert

from speeder.utils import *

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CROP_RATIO = 224/256

def get_ffcv_datasets(cfg):
    loaders = {
        'train': None,
        'val': None,
        'test': None,
    }

    for name in ['train', 'val', 'test']:
        path = os.path.join(cfg.data_path, cfg.dataset, f'{name}.beton')

        if not os.path.exists(path):
            pr(f'PATH: {path} DOES NOT EXIST. SKIPPING.')
            continue

        image_pipeline = []
        if name == 'train':
            image_pipeline += [
                RandomResizedCropRGBImageDecoder((224, 224)),
                RandomHorizontalFlip(),
            ]
        else:
            image_pipeline += [
                CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO),
            ]

        image_pipeline += [
            ToTensor(),
            # ToDevice(torch.device(DEVICE), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            # ToDevice(torch.device(DEVICE), non_blocking=True),
        ]

        order = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(
            path,
            batch_size=cfg.batch_size,
            num_workers=cfg.data_workers,
            order=order,
            drop_last=(name == 'train'),
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            },
            # os_cache=True,
        )
        
    return loaders

def get_clean_datasets(cfg):
    loaders = {
        'train': None,
        'val': None,
        'test': None,
    }

    for name in ['train', 'val', 'test']:
        path = os.path.join(cfg.data_path, cfg.dataset, f'{name}')

        if not os.path.exists(path):
            pr(f'PATH: {path} DOES NOT EXIST. SKIPPING.')
            continue

        transform = []
        if name == 'train':
            transform += [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform += [
                transforms.CenterCrop(224),
            ]

        transform += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            ),
        ]

        transform = transforms.Compose(transform)

        shuffle = True if name == 'train' else False

        ds = datasets.ImageFolder(
            path,
            transform=transform,
        )
        loaders[name] = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.data_workers,
            shuffle=shuffle,
        )

    return loaders
