import os
from time import time
from dotenv import load_dotenv

import hydra
from omegaconf import OmegaConf

from ray import tune # This is used for instantiating tune parameters from hydra
from ray.tune import Tuner, TuneConfig
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import WandbLoggerCallback

from speeder.utils import *
from speeder.models import *
from speeder.data_utils import *
from speeder.train_utils import *

@hydra.main(version_base=None, config_path='../configs', config_name='train_cfg')
def main(cfg):
    # Environment organization

    load_dotenv()

    os.remove(f'{os.path.splitext(os.path.basename(__file__))[0]}.log')
    os.umask(0)

    set_seeds(0)

    if cfg.overrides: cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.overrides))
    cfg = DotDict(cfg)

    use_gpu = len(set_gpus(cfg.gpu_ids)) > 0

    assert cfg.exp_name and cfg.run_name, 'Must specify experiment and run name!'

    exp_name = f'exp-{cfg.exp_name}' if not cfg.debug else 'exp-debug'
    
    run_id = int(time())
    run_name = f'{run_id}-run-{cfg.run_name}'

    wandb_api_key = os.getenv('WANDB_API_KEY', None)
    callbacks = [WandbLoggerCallback(
        entity=cfg.team_name, # User/organization
        project=cfg.project_name, # Research project
        group=exp_name, # Experiment
        job_type=run_name, # Run
        api_key=wandb_api_key,
    )] if wandb_api_key and not cfg.debug else None

    # Initialize tune parameters

    for k,v in cfg.items():
        if isinstance(v, str) and v.startswith('tune.'):
            cfg[k] = eval(v)

    # Start ray
    ray.init(ignore_reinit_error=True, dashboard_host='0.0.0.0')

    ds = get_global_datasets(cfg)

    # A scaling config applies to a SINGLE trainable.
    # Each tune trial uses its own trainable, so the less
    # resources per trainable means MORE trials in parallel.
    scaling_cfg = ScalingConfig(
        num_workers=cfg.train_workers,
        use_gpu=use_gpu,
        placement_strategy='PACK',
        trainer_resources=None,
        resources_per_worker={
            'CPU': cfg.cpus_per_train_worker,
            'GPU': cfg.gpus_per_train_worker
        }
    )

    param_cfg = {
        'train_loop_config': cfg
    }

    tune_cfg = TuneConfig(
        metric='loss', # TODO
        mode='min',
        search_alg=None, # TODO
        scheduler=None, # TODO
        num_samples=8, # TODO
        reuse_actors=True,
        trial_name_creator=None, # TODO
        trial_dirname_creator=None, # TODO
    )

    run_cfg = RunConfig(
        name=run_name,
        storage_path=cfg.runs_path,
        failure_config=None,
        checkpoint_config=None,
        stop=None,
        log_to_file=True,
        callbacks=callbacks
    )

    trainer = TorchTrainer(
        train_loop_per_worker,
        scaling_config=scaling_cfg,
        datasets=ds
    )

    tuner = Tuner(
        trainer.as_trainable(),
        param_space=param_cfg,
        tune_config=tune_cfg,
        run_config=run_cfg,
    )
    
    tuner.fit()

if __name__ == '__main__':
    main()