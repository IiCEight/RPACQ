# RPACQ

This is the official implementation of paper "Reliable Prototype Alignment via Curated Queues
for Cross-Subject EEG Emotion Recognition".

## Supported Datasets

| Dataset  | Subjects | Electrodes | Classes | Notes                         |
|----------|----------|------------|---------|-------------------------------|
| SEED     | 15       | 62         | 3       | 3 sessions                    |
| SEED-IV  | 15       | 62         | 4       | 3 sessions                    |
| DEAP     | 32       | 32         | 2       | valence / arousal via `--label-type` |

## Requirements

- Python ≥ 3.12
- PyTorch ~2.8
- See `pyproject.toml` for full dependency list

## Installation

```bash
uv sync
```

## Usage

```bash
python main.py --dataset SEED --dataset-path /path/to/SEED

python main.py --dataset SEED_IV --dataset-path /path/to/SEED_IV

python main.py --dataset DEAP --dataset-path /path/to/DEAP

# Key options
python main.py --help
```

## Evaluation Protocol

Leave-One-Subject-Out (LOSO): each subject is held out as the target domain in turn.
Final metric: mean accuracy across subjects over the best session.
