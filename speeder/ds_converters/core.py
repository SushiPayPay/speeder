import os
import numpy as np
from torchvision.datasets import CIFAR100
from PIL import Image
import ray
from ray.experimental.tqdm_ray import tqdm
from time import time

s = int(time())

ray.init()

root_dir = '/data/clean/cifar100'
train_dir = os.path.join(root_dir, 'train')
test_dir = os.path.join(root_dir, 'test')

train_set = CIFAR100(root='/data/clean/cifar100-dirty', train=True, download=True)
test_set = CIFAR100(root='/data/clean/cifar100-dirty', train=False, download=True)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in train_set.classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

@ray.remote
def save_images(dataset, data_dir, indices, bar):
    for idx in indices:
        img, label = dataset[idx]
        class_name = dataset.classes[label]
        img = Image.fromarray(np.array(img))
        img.save(os.path.join(data_dir, class_name, f"{idx}.png"))
        bar.update.remote(1)

def chunk_indices(length, num_workers):
    chunk_size = (length + num_workers - 1) // num_workers  # ceiling division
    return [range(i, min(i + chunk_size, length)) for i in range(0, length, chunk_size)]

num_workers = 32

ds_train_id = ray.put(train_set)
ds_test_id = ray.put(test_set)

bar = ray.remote(tqdm).remote(total=len(train_set))
train_indices = chunk_indices(len(train_set), num_workers)
train_tasks = [save_images.remote(ds_train_id, train_dir, indices, bar) for indices in train_indices]
ray.get(train_tasks)
#bar.close.remote()

bar = ray.remote(tqdm).remote(total=len(test_set))
test_indices = chunk_indices(len(test_set), num_workers)
test_tasks = [save_images.remote(ds_test_id, test_dir, indices, bar) for indices in test_indices]
ray.get(test_tasks)
#bar.close.remote()

# Shut down Ray
ray.shutdown()

e =  int(time())

print(f'Total Time: {e-s}s')
