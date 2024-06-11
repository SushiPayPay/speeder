import os
from time import time
from dotenv import load_dotenv

import hydra

from ray.tune import Tuner, TuneConfig, grid_search
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import WandbLoggerCallback

from speeder.utils import *
from speeder.models import *
from speeder.data_utils import *
from speeder.train_utils import *

@hydra.main(version_base=None, config_path='../configs', config_name='train_cfg')
def main(cfg):
    os.remove(f'{os.path.splitext(os.path.basename(__file__))[0]}.log')
    os.umask(0)

    set_seeds(0)
    use_gpu = len(set_gpus(cfg.gpus)) > 0

    load_dotenv()

    ray.init(ignore_reinit_error=True)

    cfg = DotDict(cfg)

    assert cfg.exp_name and cfg.run_name, 'Must specify experiment and run name!'

    exp_name = f'exp-{cfg.exp_name}' if not cfg.debug else 'exp-debug'
    
    run_id = int(time())
    run_name = f'{run_id}-run-{cfg.run_name}'

    wandb_api_key = os.getenv('WANDB_API_KEY', None)
    callbacks = [WandbLoggerCallback(
        entity=cfg.team_name, # user/organization
        project=cfg.project_name, # research project
        group=exp_name, # experiment
        job_type=run_name, # run
        api_key=wandb_api_key,
    )] if wandb_api_key and not cfg.debug else None

    ds = get_global_datasets(cfg)

    scaling_cfg = ScalingConfig(
        num_workers=cfg.train_workers,
        use_gpu=use_gpu,
        placement_strategy='PACK',
        trainer_resources={
            'CPU': cfg.cpus_per_train_worker,
            'GPU': cfg.gpus_per_train_worker
        }
    )

    param_cfg = {
        'train_loop_config': {
            'lr': grid_search([0.001, 0.01, 0.1])
        }
    }

    tune_cfg = TuneConfig(
        metric='loss', # TODO
        mode='min',
        search_alg=None,
        scheduler=None,
        num_samples=1, # TODO
        reuse_actors=False,
        trial_name_creator=None, # TODO
        trial_dirname_creator=None # TODO
    )

    run_cfg = RunConfig(
        name=run_name,
        storage_path=cfg.runs_path,
        failure_config=None,
        checkpoint_config=None,
        stop=None,
        log_to_file=False,
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