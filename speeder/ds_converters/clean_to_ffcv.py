import os
from dotenv import load_dotenv

import hydra
from omegaconf import OmegaConf

import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from speeder.utils import *

def clean_to_ffcv(cfg):
    datasets = {
        'train': torchvision.datasets.ImageFolder(os.path.join(cfg.input_dir, 'train')),
        'test': torchvision.datasets.ImageFolder(os.path.join(cfg.input_dir, 'test'))
    }

    for (name, ds) in datasets.items():
        writer = DatasetWriter(os.path.join(cfg.output_dir, f'{name}.beton'), {
            'image': RGBImageField(write_mode=cfg.write_mode, jpeg_quality=cfg.jpeg_quality),
            'label': IntField()
        }, num_workers=cfg.num_workers)
        writer.from_indexed_dataset(ds, chunksize=cfg.chunk_size)

@hydra.main(version_base=None, config_path='../../configs', config_name='clean_to_ffcv_cfg')
def main(cfg):
    os.remove(f'{os.path.splitext(os.path.basename(__file__))[0]}.log')
    load_dotenv()

    if cfg.overrides: cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.overrides))
    cfg = DotDict(cfg)

    save_cfg = DotDict()
    save_cfg.write_mode = cfg.write_mode
    if cfg.write_mode == 'jpg':
        save_cfg.jpeg_quality = cfg.jpeg_quality

    os.makedirs(cfg.output_dir, exist_ok=True)
    OmegaConf.save(dict(save_cfg), os.path.join(cfg.output_dir, 'cfg.yaml'))

    clean_to_ffcv(cfg)

if __name__ == '__main__':
    main()
