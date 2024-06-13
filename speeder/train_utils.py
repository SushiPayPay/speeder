import torch

from ray.train import get_dataset_shard, report
from ray.train.torch import enable_reproducibility, prepare_model

from speeder.utils import *
from speeder.models import *

def train_loop_per_worker(cfg):
    cfg = DotDict(cfg)
    set_seeds(0)
    enable_reproducibility()

    model = Net()
    model = prepare_model(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Get shard once at the very beginning
    # Shuffles will be limited to local shard context, but this is okay
    ds_train_shard = get_dataset_shard('train')
    ds_val_shard = get_dataset_shard('val')

    loader_cfg = {
        'batch_size': cfg.global_batch_size // cfg.train_workers,
        'prefetch_batches': cfg.prefetch_batches
    }

    for epoch in range(1, cfg.epochs+1):
        model.train()
        for batch in ds_train_shard.iter_torch_batches(**loader_cfg):
            print(batch)

        report({'loss': 5})
