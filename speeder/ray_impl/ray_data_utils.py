from PIL import Image
import io
import os

from torchvision import transforms
from torchvision.transforms import v2
import ray

from datasets import load_dataset

from speeder.utils import *

def get_ray_datasets(cfg):
    '''Returns ray data dataset dict
    '''
    
    ray.init(ignore_reinit_error=True, dashboard_host='0.0.0.0')

    map_cfg = {
        'concurrency': cfg.data_workers,
        'num_cpus': cfg.cpus_per_data_worker,
        'num_gpus': cfg.gpus_per_data_worker,
    }

    ds_name = 'cifar10'
    ds_dir = os.path.join(cfg.data_path, ds_name)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    def do_transforms(row):
        row['img'] = transform(Image.open(io.BytesIO(row['img']['bytes'])).convert('RGB'))
        return row

    ds = load_dataset('uoft-cs/cifar10', cache_dir=ds_dir)

    ds_train = (
        ray.data.from_huggingface(ds['train'])
        .map(do_transforms, **map_cfg)
        .random_shuffle()
    )

    pr('DS_TRAIN INFO')
    pr(ds_train.count())
    pr(ds_train.schema())

    ds_val = (
        ray.data.from_huggingface(ds['test'])
        .map(do_transforms, **map_cfg)
        .random_shuffle()
    )

    pr(f'DS_VAL INFO:')
    pr(ds_val.count())
    pr(ds_val.schema())

    ds = {
        'train': ds_train,
        'val': ds_val
    }

    return ds