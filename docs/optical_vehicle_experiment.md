# 5-layer scattering-medium optical inference

This experiment replays the trained `measured_tm_scatter` network on the
physical DMD/scattering-medium/camera setup used by `combined_app_v4_128.py`.
It is an inference/control program, not an additional training loop.

## Physical interpretation

The trained five-layer model reuses one measured transmission matrix at every
layer. The corresponding laboratory implementation therefore time-multiplexes
one optical path five times:

```text
input image
  -> digital preprocessing
  -> DMD complex-field hologram (phase mask 0)
  -> scattering medium
  -> I90 camera intensity
  -> digital min-max normalization
  -> DMD complex-field hologram (phase mask 1)
  -> ... repeat through phase mask 4
  -> final camera intensity / argmax position
```

The recurrence follows the preprocessing semantics stored in each checkpoint:

- For the current vehicle dataset, PNG gray values are complex-field
  amplitudes. Layer 0 therefore uses max-only normalization, does not take a
  square root, and preserves the deliberately non-zero dark background.
- Legacy checkpoints without `input_amplitude_normalization` retain their old
  min-max plus square-root behavior.
- A hidden-layer camera **intensity** is min-max normalized and used directly
  as the next displayed field amplitude. It is not square-rooted.

Every layer forms `amplitude * exp(+i * phase)`. There is no phase conjugation,
matrix transpose, random-matrix scale, or sparsification in this path.

## Relevant files

- `scripts/run_optical_vehicle_experiment.py`: command-line experiment runner.
- `donn_lab/experiment/optical_inference.py`: model-equivalent five-layer state
  machine, dark correction, QC, and localization metrics.
- `donn_lab/experiment/vehicle_data.py`: stable sample IDs plus the exact
  training-time vehicle preprocessing and row/column labels.
- `donn_lab/hardware/torch_tm_backend.py`: ideal measured-TM simulator.
- `donn_lab/hardware/v4_128_backend.py`: guarded V4 DMD/camera adapter.
- `donn_lab/experiment/artifacts.py`: atomic sample outputs and strict resume.

## Simulation first

Run one validation image through the exact five-layer recurrence:

```powershell
C:\Users\smart\miniconda3\envs\donn-lab\python.exe `
  scripts\run_optical_vehicle_experiment.py `
  --mode simulate `
  --checkpoint runs\checkpoints\base\measured_tm_scatter\vehicle\20260812-170342_vehicle-dark-intersection-measured-tm-scatter\epoch_300.pth `
  --tmatrix tm.npy `
  --split val `
  --max-samples 1 `
  --batch-size 1 `
  --save-layers all
```

Use `--output-dir` to select a stable output path. Without it, the runner makes
a timestamped directory below `runs/optical_vehicle`. Pass `--resume` only for
the same checkpoint, TM identity, data configuration, split, and processing
settings; a mismatched fingerprint is rejected.

## Hardware dry run and execution

The installed FLIR/Spinnaker `PySpin` module is in the existing Python 3.8
environment, while the training environment is Python 3.12. Use `py38` for
physical acquisition. The runner loads phase tensors and data in either
environment.

First perform a non-armed dependency/configuration check. With the current
main-guarded V4 controller this imports the software stack and exercises one
hologram encoding, but constructs no camera or DMD device object and does not
project on the DMD:

```powershell
C:\Users\smart\miniconda3\envs\py38\python.exe `
  scripts\run_optical_vehicle_experiment.py `
  --mode hardware `
  --checkpoint runs\checkpoints\base\measured_tm_scatter\vehicle\20260812-170342_vehicle-dark-intersection-measured-tm-scatter\epoch_300.pth `
  --v4-module combined_app_v4_128.py `
  --v4-root C:\Users\smart\Documents\TMCalib `
  --dll-parent C:\Users\smart\Documents `
  --preflight-only
```

Only after the optical path, trigger cable, polarization selection, and target
device are ready, add `--arm-hardware`. Start with one sample:

```powershell
C:\Users\smart\miniconda3\envs\py38\python.exe `
  scripts\run_optical_vehicle_experiment.py `
  --mode hardware `
  --arm-hardware `
  --checkpoint runs\checkpoints\base\measured_tm_scatter\vehicle\20260812-170342_vehicle-dark-intersection-measured-tm-scatter\epoch_300.pth `
  --v4-module combined_app_v4_128.py `
  --v4-root C:\Users\smart\Documents\TMCalib `
  --dll-parent C:\Users\smart\Documents `
  --split val `
  --max-samples 1 `
  --batch-size 1 `
  --dark-frames 16 `
  --save-layers all
```

If more than one DMD is online, pass its exact name with `--dmd-device`.
Hardware acquisition uses the settings established by the V4 code: a central
512x512 active DMD region (128x128 logical modes, 4x4 superpixels), 2000 us DMD
picture time, a 1500 us camera exposure, `Polarized8`, I90 quadrant extraction,
and a 128x128 output ROI.

Hardware mode is deliberately inert without `--arm-hardware`. A batch is
accepted only when its pattern shapes and camera frames are valid and frame IDs
remain consecutive. Timeout or frame-gap failures stop/retry the complete
layer transaction; they are never silently inserted into the recurrence.
The default hardware quality gate also requires at least five native 8-bit
camera codes of dynamic range. Frames containing only 0/1 dark noise abort the
run instead of producing meaningless localization metrics.

## Outputs

Each run contains:

- `run.json`: immutable experiment fingerprint and resolved metadata.
- `calibration/dark.npy`: dark frame used for correction, when requested.
- `manifest.csv`: one row per completed sample.
- `summary.json`: aggregate localization and camera-quality metrics.
- `samples/<sample-id>/`: exact input, final intensity, prediction, optional
  per-layer arrays, and `DONE.json` written only after the sample is complete.

The final prediction and target use `(row, column)`, while the source CSV stores
`(center_x, center_y)`. The runner preserves the training loader conversion and
camera/TM C-order flattening; it does not silently transpose or flip the ROI.

## Before a full validation run

1. Run the simulator on one image and then a small batch.
2. Confirm a black pattern produces a valid unsaturated dark acquisition.
3. Run one physical sample and inspect all five saved layers.
4. Verify that the ROI orientation matches the TM calibration with known focus
   points (center and corners).
5. Check saturation and dynamic-range fields in `manifest.csv`.
6. Increase to a small fixed subset before attempting all 3,200 validation
   samples.

The ideal TM simulator proves software/model equivalence. It cannot compensate
for TM drift, binary-hologram diffraction efficiency, misalignment, camera
clipping, or an ROI/polarization mismatch; those remain physical calibration
conditions and are intentionally surfaced instead of hidden by automatic
exposure changes.

For GPUs that cannot cache the roughly 2 GiB complex64 matrix, add
`--stream-tm --tm-chunk-rows 512`; this trades speed for lower device memory.

## Linked retraining and hardware test

After replacing `tm.npy`, use the linked workflow instead of manually copying
an old checkpoint. It validates and fingerprints the new matrix, starts a fresh
five-layer training run in the `donn-lab` environment, discovers the checkpoint
created by that exact run, applies a simulation accuracy gate, then uses the
`py38` environment for hardware preflight and (only when explicitly armed) a
one-sample physical test.

The safe default performs training, simulation, and hardware software preflight
but does not initialize a camera or DMD:

```powershell
C:\Users\smart\miniconda3\envs\donn-lab\python.exe `
  scripts\train_and_test_optical_vehicle.py
```

To permit the final hardware stage, add `--arm-hardware`. The workflow still
pauses after training and asks the operator to type `ARM`, so the laboratory
state can be checked immediately before projection:

```powershell
C:\Users\smart\miniconda3\envs\donn-lab\python.exe `
  scripts\train_and_test_optical_vehicle.py `
  --arm-hardware
```

For a fast software/hardware integration check, train only 10 epochs and use
the first training-set sample for both ideal-TM simulation and the one-sample
hardware replay:

```powershell
C:\Users\smart\miniconda3\envs\donn-lab\python.exe `
  scripts\train_and_test_optical_vehicle.py `
  --epochs 10 `
  --inference-split train `
  --sim-samples 1 `
  --hardware-samples 1 `
  --allow-poor-simulation `
  --arm-hardware
```

`--allow-poor-simulation` is appropriate only for this guarded integration
check: it reports the simulated localization error but does not prevent the
explicitly armed one-sample hardware run when the 10-epoch model is still
under-trained. The resulting checkpoint and artifacts are debug outputs, not
a replacement for the full training/validation run.

Use `--yes` together with `--arm-hardware` only for intentionally unattended
operation. A dry run prints every planned command without scanning the full TM,
training, opening a backend, or creating a run directory:

```powershell
C:\Users\smart\miniconda3\envs\donn-lab\python.exe `
  scripts\train_and_test_optical_vehicle.py --dry-run
```

Each invocation gets an isolated directory below `runs/linked_optical_vehicle`
containing `workflow.json`, subprocess logs, training checkpoints, simulation
results, and optional hardware artifacts. Training or test failure stops the
chain. The TM is checked again after training; a changed file refuses to
continue. To test an already-trained checkpoint without starting training, use
`--skip-training --checkpoint <path>`.
