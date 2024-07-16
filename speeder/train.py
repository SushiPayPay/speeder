import os
from pathlib import Path
from time import time
from dotenv import load_dotenv

import hydra
from omegaconf import OmegaConf

from lightning.pytorch.loggers import WandbLogger
import lightning as L

from speeder.utils import *
from speeder.data_utils import *
from speeder.train_utils import *

@hydra.main(version_base=None, config_path='../configs', config_name='train_cfg')
def main(cfg):
    Path(f'{os.path.splitext(os.path.basename(__file__))[0]}.log').unlink(missing_ok=True)
    load_dotenv()

    os.umask(0)
    set_seeds(0)

    if cfg.overrides: cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.overrides))
    cfg = DotDict(cfg)

    assert cfg.exp_name and cfg.run_name, 'Must specify experiment and run name!'

    exp_name = f'exp-{cfg.exp_name}' if not cfg.debug else 'exp-debug'
    run_name = f'run-{cfg.run_name}-{int(time())}'

    output_path = os.path.join(cfg.runs_path, exp_name, run_name)
    os.makedirs(output_path, exist_ok=True)
    OmegaConf.save(dict(cfg), os.path.join(output_path, 'cfg.yaml'))

    torch.set_float32_matmul_precision('medium')

    wandb_logger = WandbLogger(
        entity=cfg.team_name,
        project=cfg.exp_name,
        name=cfg.run_name,
        config=cfg,
        dir=output_path
    )

    trainer = L.Trainer(
        devices=cfg.gpu_ids,
        strategy='ddp',
        # strategy="deepspeed_stage_2",
        precision='16-mixed',
        logger=wandb_logger,
        max_epochs=cfg.epochs
    )

    model = LightningModel()

    loaders = None
    if 'ffcv' in cfg.dataset:
        loaders = get_ffcv_datasets(cfg)
    else:
        loaders = get_clean_datasets(cfg)

    trainer.fit(
        model=model,
        train_dataloaders=loaders['train'],
        val_dataloaders=loaders['test'],
    )

if __name__ == '__main__':
    main()
