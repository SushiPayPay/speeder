from datasets import load_dataset
import ray
import numpy as np
from torchvision import transforms
from PIL import Image
import io
from pprint import pprint
import time
from torchvision.transforms import v2
import hydra
import os

from speeder.utils import *
from speeder.models import *

def get_global_datasets(cfg):
    '''Returns train and val ray dataset objects
    '''

    map_cfg = {
        'concurrency': cfg.num_data_workers,
        'num_cpu': cfg.num_cpus_per_data_worker,
        'num_gpu': cfg.num_gpus_per_data_worker,
    }

    ds_dir = '/home/data/huggingface/tiny-imagenet'

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
        ray.data.from_huggingface(ds['validation'])
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

def train_loop_per_worker(cfg):
    set_seeds(0)

    model = Net()
    model = ray.train.torch.prepare_model(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Get shard once at the very beginning
    # Shuffles will be limited to local shard context, but this is okay
    ds_train_shard = ray.train.get_dataset_shard('train')
    ds_val_shard = ray.train.get_dataset_shard('val')

    loader_cfg = {
        'batch_size': cfg.global_batch_size // cfg.num_train_workers,
        'prefetch_batches': cfg.prefetch_batches
    }

    for epoch in range(1, cfg.epochs+1):
        model.train()
        for batch in ds_train_shard.iter_torch_batches(**loader_cfg):
            pass

@hydra.main(version_base=None, config_path='../configs', config_name='train_cfg')
def main(cfg):
    os.remove(f'{os.path.splitext(os.path.basename(__file__))[0]}.log')
    os.umask(0)

    ray.init()

    cfg = DotDict(cfg)

    run_cfg = ray.train.RunConfig()

    scaling_cfg = ray.train.ScalingConfig(
        num_workers=cfg.num_train_workers,
        use_gpu=cfg.use_gpu,
        trainer_resources={
            'CPU': cfg.cpus_per_train_worker,
            'GPU': cfg.gpus_per_train_worker
        }
    )

    ds = get_global_datasets(cfg)

    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker,
        train_loop_config=cfg,
        run_config=run_cfg,
        scaling_config=scaling_cfg,
        datasets=ds
    )
    result = trainer.fit()

if __name__ == '__main__':
    main()