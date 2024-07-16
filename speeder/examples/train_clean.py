import numpy as np
from tqdm import tqdm
import os

from time import time

import torch
import torchvision
import torchvision.transforms as transforms

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision

from speeder.models import *
from speeder.utils import *

PATH = '/data/clean/tiny-imagenet'
DEVICE = 'cuda:0'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

'''
CIFAR10:
    train: randomhorizontalflip

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
'''

def get_dataloaders():
    loaders = {}
    for name in ['train', 'test']:
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

        batch_size = 512
        num_workers = 8
        shuffle = True if name == 'train' else False

        ds = torchvision.datasets.ImageFolder(
            os.path.join(PATH, name),
            transform=transform,
        )
        loaders[name] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
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
            X,y = X.to(DEVICE),y.to(DEVICE)
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
                X,y = X.to(DEVICE),y.to(DEVICE)
                with autocast():
                    out = model(X)
                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%")

def main():
    s = int(time())

    loaders = get_dataloaders()
    model = RN50(num_classes=1000).to(torch.device(DEVICE))

    train(model, loaders)

    e = int(time())

if __name__ == "__main__":
    main()
