import numpy as np
from tqdm import tqdm
import os

from time import time

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from speeder.models import *

DEVICE = 'cuda:0'

paths = {
    'train': '/data/ffcv/cifar10/train.beton',
    'test': '/data/ffcv/cifar10/test.beton'
}


CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
loaders = {}

def get_dataloaders():
    for name in ['train', 'test']:
        label_pipeline = [IntDecoder(), ToTensor(), ToDevice(torch.device('cuda:0')), Squeeze()]
        image_pipeline = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(torch.device(DEVICE), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], batch_size=64, num_workers=8,
                                order=ordering, drop_last=(name == 'train'),
                                pipelines={'image': image_pipeline, 'label': label_pipeline})
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
    model = RN50(num_classes=10).to(torch.device(DEVICE))

    train(model, loaders)

    e = int(time())

    print(f'Total Time: {e-s}s')
