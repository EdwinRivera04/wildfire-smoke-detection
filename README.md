# Wildfire Smoke Detection

YOLOv8m-based early wildfire smoke and fire detection using the D-Fire dataset.

## Team
- **Edwin** — data pipeline (`src/data/`, `data/mock/`, `src/utils/paths.py`)
- **Lyla** — training (`src/train.py`, `configs/`, Colab notebook)
- **Ann** — evaluation & demo (`src/evaluate.py`, `src/demo.py`, `src/data/augment.py`)

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start (no D-Fire download needed)

```bash
make mock          # validate mock dataset
make train-mock    # train for 1 epoch on mock data (<60s)
make evaluate-mock # evaluate with mock predictions
make demo-mock     # run inference on mock images
make test          # run all tests
```

## Full Pipeline

```bash
python src/data/prepare_dataset.py  # download + split D-Fire
make train-baseline                  # 25-epoch baseline run
make train-improved                  # 15-epoch improved run
make evaluate                        # compute all metrics
make demo                            # generate annotated JPEGs
```

## Environments

| Setting   | Local (M4 MPS) | Colab (T4)  |
|-----------|---------------|-------------|
| `device`  | mps           | cuda        |
| `workers` | 0             | 4           |
| `imgsz`   | 416           | 640         |
| `batch`   | 32            | 16          |

Configs in the repo always store MPS-safe values. The Colab notebook patches them at runtime.

## Branches

| Branch          | Owner | Contents                          |
|-----------------|-------|-----------------------------------|
| `main`          | all   | stable — `make test` must pass    |
| `edwin/data`    | Edwin | data pipeline + mock data         |
| `lyla/training` | Lyla  | configs, Makefile, train.py       |
| `ann/evaluation`| Ann   | augment, evaluate, demo, tests    |
