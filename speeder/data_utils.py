from PIL import Image
import io
import os

from torchvision import transforms
from torchvision.transforms import v2
import ray

from datasets import load_dataset

from speeder.utils import *

def get_global_datasets(cfg):
    '''Returns train and val ray dataset objects
    '''
    
    ray.init(ignore_reinit_error=True)

    map_cfg = {
        'concurrency': cfg.data_workers,
        'num_cpus': cfg.cpus_per_data_worker,
        'num_gpus': cfg.gpus_per_data_worker,
    }

    ds_name = 'tiny-imagenet'
    ds_dir = os.path.join(cfg.data_path, ds_name)

    transform = transforms.Compose([
        v2.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    def do_transforms(row):
        row['image'] = transform(Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB'))
        return row

    ds = load_dataset('zh-plus/tiny-imagenet', cache_dir=ds_dir)

    ds_train = (
        ray.data.from_huggingface(ds['train'])
        .map(do_transforms, **map_cfg)
    )

    print(f'{sty.RED}DS_TRAIN INFO:{sty.RESET}')
    print(ds_train.count())
    print(ds_train.schema())

    ds_val = (
        ray.data.from_huggingface(ds['valid'])
        .map(do_transforms, **map_cfg)
    )

    print(f'{sty.RED}DS_VAL INFO:{sty.RESET}')
    print(ds_val.count())
    print(ds_val.schema())

    ds = {
        'train': ds_train,
        'val': ds_val
    }

    return ds