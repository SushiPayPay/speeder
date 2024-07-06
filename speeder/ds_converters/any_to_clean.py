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

# mnist
# celeba
# cifar10
# cifar100
# tiny-imagenet
# imagenet
# cinic10
# stl10

import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from multiprocessing import Process, Queue, Value
from tqdm import tqdm
from time import sleep, time

s = int(time())

INPUT_DIR = '/data/clean/mnist-dirty'
OUTPUT_DIR = '/data/clean/mnist'

# Define paths
train_dir = os.path.join(OUTPUT_DIR, 'train')
test_dir = os.path.join(OUTPUT_DIR, 'test')

# Create necessary directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Download the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root=INPUT_DIR, train=True, download=True, transform=transform)
test_dataset = MNIST(root=INPUT_DIR, train=False, download=True, transform=transform)

def save_image_worker(queue, done_count):
    while True:
        args = queue.get()
        if args is None:
            break
        img, label, idx, data_dir = args
        label_dir = os.path.join(data_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        img_path = os.path.join(label_dir, f'{idx}.png')
        img = transforms.ToPILImage()(img)
        img.save(img_path)
        with done_count.get_lock():
            done_count.value += 1

def process_dataset(dataset, data_dir, num_workers=1):
    queue = Queue()
    done_count = Value('i', 0)

    processes = [Process(target=save_image_worker, args=(queue, done_count)) for _ in range(num_workers)]
    for p in processes:
        p.start()

    for idx, (img, label) in enumerate(dataset):
        queue.put((img, label, idx, data_dir))

    for _ in range(num_workers):
        queue.put(None)

    progress = tqdm(total=len(dataset))
    previous = 0
    while previous != len(dataset):
        with done_count.get_lock():
            current = done_count.value
        progress.update(current - previous)
        previous = current
        sleep(0.01)
    progress.close()

    for p in processes:
        p.join()

# Process and save train and test datasets in parallel
process_dataset(train_dataset, train_dir, num_workers=32)
process_dataset(test_dataset, test_dir, num_workers=32)

print('MNIST dataset has been saved in ImageFolder format.')

e = int(time())

print(f'Time taken: {e-s}s')





