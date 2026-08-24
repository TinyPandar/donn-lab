# DONN Lab

Lightweight lab-side code for measured transmission-matrix DONN experiments.

This repository is split out from the larger research workspace so it can move
to a lab server without old datasets, checkpoints, logs, or visualization
outputs. The workflow is file-oriented:

```text
measure H0 -> train phase masks with H0 -> measure H1 -> evaluate H0/H1
           -> compare drift -> apply phase correction -> evaluate again
```

## Repository Layout

- `donn_lab/`: reusable lab interfaces for TM IO, drift analysis, checkpoint phase correction, and hardware protocols.
- `models/optical/measured_tm_network.py`: measured-TM optical forward model with trainable phase masks.
- `config/`, `core/`, `data/`, `engine/`, `models/`, `pipelines/`, `registry/`: minimal training stack copied from the research repo.
- `scripts/`: thin command-line wrappers for the measured-TM workflow.
- `configs/vehicle_dark_intersection_measured_tm_scatter_template.yaml`: starting config for the 5-layer dark-intersection measured-TM run.
- `docs/stage0_measured_tm_interface.md`: detailed stage-0 file contract and command examples.

## Data Policy

Keep these outside git:

- datasets
- measured matrices: `H0.npy`, `H1.npy`, raw memmap files
- checkpoints: `*.pth`, `*.pt`
- tensorboard logs
- generated reports, figures, and videos

Recommended server paths:

```text
/data/donn/datasets/
/data/donn/tm/
/data/donn/runs/
/data/donn/reports/
```

## Environment

Create an environment from `environment.yml`, or install the equivalent
packages manually. The CUDA/PyTorch line may need to match the lab server.

```bash
conda env create -f environment.yml
conda activate donn-lab
```

## Print The Stage-0 Workflow

```bash
python scripts/plan_measured_tm_workflow.py \
  --config configs/vehicle_dark_intersection_measured_tm_scatter_template.yaml \
  --h0 /data/donn/tm/run_001/H0.npy \
  --h1 /data/donn/tm/run_001/H1.npy \
  --run-dir /data/donn/runs/run_001 \
  --report-dir /data/donn/reports/run_001
```

## Train With A Measured Matrix

```bash
python scripts/train_measured_tm.py \
  --config configs/vehicle_dark_intersection_measured_tm_scatter_template.yaml \
  --tmatrix_path /data/donn/tm/run_001/H0.npy \
  --num_layers 5
```

## MNIST Classification With A 32x24 Input

The MNIST classification pipeline keeps the optical output as a `128x128`
intensity map and reads ten fixed detector regions arranged in a `2x5` grid.
MNIST is resized to `24x24` without distortion and zero-padded on the left and
right to produce the `24x32` (`H x W`) optical input. The legacy MNIST
class-to-single-pixel coordinate mode remains available through
`data.mnist_target_mode: coord`.

```bash
python train.py \
  --config configs/mnist_32x24_measured_tm_classification_template.yaml \
  --tmatrix_path /data/donn/tm/run_001/H0.npy
```

The template expects a complex matrix with shape `[16384, 768]`, corresponding
to `128x128` camera pixels and `32x24` input modes. Run the software-only
classification check with:

```bash
python tests/smoke_mnist_classification.py
```

## Evaluate With H0 Or H1

```bash
python scripts/evaluate_measured_tm.py \
  --run-dir /data/donn/runs/run_001 \
  --tmatrix-path /data/donn/tm/run_001/H1.npy \
  --out /data/donn/reports/run_001/eval_h1.csv
```

## Local Smoke Test

```bash
python tests/smoke_measured_tm.py
```

