import torch

from ray.train import get_dataset_shard, report
from ray.train.torch import enable_reproducibility, prepare_model
from ray.experimental.tqdm_ray import tqdm

from speeder.utils import *
from speeder.models import *

def train_loop_per_worker(cfg):
    cfg = DotDict(cfg)
    set_seeds(0)
    enable_reproducibility()

    model = RN50()
    model = prepare_model(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Get shard once at the very beginning
    # Shuffles will be limited to local shard context, but this is okay
    ds_train = get_dataset_shard('train')
    ds_val = get_dataset_shard('val')

    loader_cfg = {
        'batch_size': cfg.batch_size // cfg.train_workers,
        'prefetch_batches': cfg.prefetch_batches
    }

    dl_train = ds_train.iter_torch_batches(**loader_cfg)
    dl_val = ds_val.iter_torch_batches(**loader_cfg)

    for epoch in range(1, cfg.epochs+1):
        model.train()
        for batch in tqdm(dl_train, desc=f"Train Epoch {epoch}"):
            X, y = batch['image'], batch['label']
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"Test Epoch {epoch}"):
                X, y = batch['image'], batch['label']
                pred = model(X)
                loss = loss_fn(pred, y)

                test_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        test_loss /= num_total
        accuracy = num_correct / num_total

        report(metrics={"loss": test_loss, "accuracy": accuracy})
