# Stage 0 measured-TM software interface

This stage keeps the hardware measurement code separate from training, and
defines the file-level contract between them.

The reusable code is now organized under `donn_lab/`; command-line scripts in
`scripts/` are intentionally thin wrappers around that package.

## Matrix format

- Store the measured transmission matrix as `complex64` `.npy` or `.npz`.
- Default layout is `[N_out, N_in]`, where rows are camera ROI pixels and columns are input SLM/DMD modes.
- For the planned 128 x 128 input to 128 x 128 camera ROI setup:
  - `N_in = 16384`
  - `N_out = 16384`
  - matrix shape is `16384 x 16384`
  - `complex64` storage is about 2 GiB per matrix
- Raw memmap files are supported if `--shape 16384,16384` is provided.

## Train with H0

```bash
/home/limingfei/miniforge3/envs/speckle/bin/python scripts/train_measured_tm.py \
  --config configs/vehicle_dark_intersection_measured_tm_scatter_template.yaml \
  --tmatrix_path /path/to/H0.npy \
  --num_layers 5
```

The measured matrix is loaded as a fixed buffer and is not saved inside each
checkpoint. Checkpoints keep the trainable phase masks and optimizer state.

## Print the closed-loop command sequence

```bash
/home/limingfei/miniforge3/envs/speckle/bin/python scripts/plan_measured_tm_workflow.py \
  --config configs/vehicle_dark_intersection_measured_tm_scatter_template.yaml \
  --h0 /path/to/H0.npy \
  --h1 /path/to/H1.npy \
  --run-dir /path/to/run_dir \
  --report-dir reports/measured_tm_closed_loop
```

## Evaluate on a chosen matrix

```bash
/home/limingfei/miniforge3/envs/speckle/bin/python scripts/evaluate_measured_tm.py \
  --run-dir logs/base/measured_tm_scatter/vehicle/<run_name> \
  --tmatrix-path /path/to/H1.npy \
  --out reports/measured_tm_eval/h1_metrics.csv
```

Use the same command with `H0.npy` and `H1.npy` to measure how much the test
metric changes after the hardware is remeasured.

## Compare H0/H1 drift

```bash
/home/limingfei/miniforge3/envs/speckle/bin/python scripts/compare_tm_drift.py \
  --h0 /path/to/H0.npy \
  --h1 /path/to/H1.npy \
  --out-dir reports/tm_drift_compare
```

This writes row/column gain, phase, and correlation summaries. Row phase is
mostly a gauge term for intensity-only evaluation; column phase is the first
candidate correction for the input phase masks.

## Apply column phase correction

```bash
/home/limingfei/miniforge3/envs/speckle/bin/python scripts/apply_tm_correction.py \
  --checkpoint logs/base/measured_tm_scatter/vehicle/<run_name>/epoch_300.pth \
  --h0 /path/to/H0.npy \
  --h1 /path/to/H1.npy \
  --out-dir reports/tm_phase_correction
```

The default assumes `H1 ~= R * H0 * D`, estimates the input-column phase in
`D`, and updates each `phases.<idx>` tensor as `phase - delta`. The script runs
a few alternating row/column phase-fit passes by default, which is useful when
the complex matrix is reconstructed with an output-row phase gauge. Add
`--no-row-align` if H0/H1 come from an interferometric measurement with a stable
output phase reference.

It does not attempt to compensate amplitude/gain drift, because a phase-only
mask cannot directly implement arbitrary column gain.
