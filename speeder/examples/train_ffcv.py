import numpy as np
from tqdm import tqdm
import os

from time import time

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torchvision

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

from speeder.models import *
from speeder.utils import *

PATH = '/data/clean/tiny-imagenet-ffcv'
DEVICE = 'cuda:1'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CROP_RATIO = 224/256

def get_dataloaders():
    loaders = {}
    for name in ['train', 'test']:
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
            ToDevice(torch.device(DEVICE), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(DEVICE), non_blocking=True),
        ]

        batch_size = 512
        num_workers = 8
        order = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(
            os.path.join(PATH, f'{name}.beton'),
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            drop_last=(name == 'train'),
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            }
        )
        
    return loaders

def train(model, loaders):
    lr = 0.01
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    loss_fn = CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(10):
        model.train()
        for X,y in tqdm(loaders['train']):
            optimizer.zero_grad()
            with autocast():
                out = model(X)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in loaders['test']:
                with autocast():
                    out = model(X)
                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")

if __name__ == "__main__":
    s = int(time())

    loaders = get_dataloaders()
    model = RN50(num_classes=200).to(torch.device(DEVICE))

    train(model, loaders)

    e = int(time())

    print(f'Total Time: {e-s}s')
