# Speeder: Accelerated ImageNet with Ray

## Data Pipeline Parallelism (DPP)
Speeder uses RayData to do distributed data preprocessing on the CPU at training time with data prefetching

## Distributed Data Parallelism (DDP)
Speeder uses RayTrain to do distributed data parallelism to train models on multiple GPUs at once

# Quickstart

```
conda env create -y -n speeder -f env.yaml
poetry install
python speeder/train.py
```

You can modify parameters directly in `configs/train_cfg.yaml`

For launching runs, you can specify a parameter override path to override select defaults like so:
```
python speeder/train.py overrides=<path_to_your_overrides_yaml>
```
See the `overrides` directory for examples