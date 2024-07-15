import hydra
from omegaconf import OmegaConf

import os
from time import time
from dotenv import load_dotenv

from PIL import Image
import numpy as np

import ray
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset

from speeder.utils import *

class TorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        x = self.ds[int(i)]
        return x['img'].convert('RGB'), int(x['fine_label'])
    
    def __len__(self):
        return len(self.ds)

@ray.remote
def save_images(ds, dir, indices, bar):
    for i in indices:
        X,y = ds[i]
        X = Image.fromarray(np.array(X))
        X.save(os.path.join(dir, str(y), f"{i}.png"))
        bar.update.remote(1)

def chunk_indices(length, num_workers):
    chunk_size = (length+num_workers-1) // num_workers
    return [range(i, min(i+chunk_size, length)) for i in range(0, length, chunk_size)]

def raw_to_clean(cfg):
    s = int(time()*1000)/1000

    # FLEX AREA <OPEN>

    ds = load_dataset('uoft-cs/cifar100', cache_dir=cfg.input_dir)

    # Make directories
    train_dir = os.path.join(cfg.output_dir, 'train')
    num_train_labels = len(ds['train'].info.features['fine_label'].names)
    for label in range(num_train_labels):
        os.makedirs(os.path.join(train_dir, str(label)), exist_ok=True)

    test_dir = os.path.join(cfg.output_dir, 'test')
    num_test_labels = len(ds['test'].info.features['fine_label'].names)
    for label in range(num_test_labels):
        os.makedirs(os.path.join(test_dir, str(label)), exist_ok=True)

    # Make torch datasets
    ds_train = TorchDataset(ds['train'])
    ds_test = TorchDataset(ds['test'])

    # FLEX AREA <CLOSE>

    # Load into ray shared object store
    ds_train_object = ray.put(ds_train)
    ds_test_object = ray.put(ds_test)

    bar_actor = ray.remote(tqdm).remote(total=len(ds_train))
    train_indices = chunk_indices(len(ds_train), cfg.num_workers)
    train_tasks = [save_images.remote(ds_train_object, train_dir, indices, bar_actor) \
                    for indices in train_indices]
    ray.get(train_tasks)
    bar_actor.close.remote()

    bar_actor = ray.remote(tqdm).remote(total=len(ds_test))
    test_indices = chunk_indices(len(ds_test), cfg.num_workers)
    test_tasks = [save_images.remote(ds_test_object, test_dir, indices, bar_actor) \
                    for indices in test_indices]
    ray.get(test_tasks)
    bar_actor.close.remote()

    e = int(time()*1000)/1000

    pl(f'Total Time: {e-s}s')

@hydra.main(version_base=None, config_path='../../configs', config_name='raw_to_clean_cfg')
def main(cfg):
    os.remove(f'{os.path.splitext(os.path.basename(__file__))[0]}.log')
    load_dotenv()

    if cfg.overrides: cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.overrides))
    cfg = DotDict(cfg)

    raw_to_clean(cfg)

if __name__ == '__main__':
    main()

'''
mnist
celeba
cifar10
cifar100
tiny-imagenet
imagenet
cinic10
stl10
'''
