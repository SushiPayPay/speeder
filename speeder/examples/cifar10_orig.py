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

DEVICE = 'cuda:1'

CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
loaders = {}

def get_dataloaders():
    for name in ['train', 'test']:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 1024

        ds = torchvision.datasets.ImageFolder(os.path.join('/data/clean/cifar10', name), transform=transform)

        loaders[name] = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8)

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

if __name__ == "__main__":
    s = int(time())

    loaders = get_dataloaders()
    model = RN50(num_classes=10).to(torch.device(DEVICE))

    train(model, loaders)

    e = int(time())

    print(f'Total Time: {e-s}s')
