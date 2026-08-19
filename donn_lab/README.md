# DONN Lab Interface

This folder is the lightweight, lab-side layer for measured transmission-matrix
experiments. It is meant to be easy to copy to a lab server without bringing
old logs, checkpoints, datasets, or visual outputs.

## What This Layer Owns

- `donn_lab/tm/io.py`: read measured matrices from `.npy`, `.npz`, or raw memmap files.
- `donn_lab/tm/drift.py`: compare H0/H1 and estimate column phase drift.
- `donn_lab/tm/checkpoint_phase.py`: apply a column phase correction to saved phase masks.
- `donn_lab/hardware/interfaces.py`: small camera/projector protocols.
- `donn_lab/hardware/mock_devices.py`: local mock devices for testing workflow code without hardware.
- `donn_lab/hardware/torch_tm_backend.py`: batched measured-TM optical simulator.
- `donn_lab/hardware/v4_128_backend.py`: guarded JUOPT/Spinnaker 128-grid adapter.
- `donn_lab/experiment/optical_inference.py`: model-faithful multi-pass camera feedback.
- `donn_lab/workflows/closed_loop.py`: command builder for the stage-0 closed loop.

The older training stack still provides datasets, losses, pipelines, logging,
and checkpoint loading. This layer only adds measured-TM and lab workflow
interfaces around that stack.

## File Contract

The hardware measurement code should write matrices like this:

```text
/data/donn/tm/<date_or_run>/H0.npy
/data/donn/tm/<date_or_run>/H1.npy
```

Default format:

- dtype: `complex64`
- layout: `[N_out, N_in]`
- `N_in = 128 * 128 = 16384`
- `N_out = 128 * 128 = 16384`

A full 128 x 128 to 128 x 128 complex64 matrix is about 2 GiB, so these files
should stay outside git.

## Stage-0 Closed Loop

1. Measure H0 with the hardware code.
2. Train phase masks with H0.
3. Measure H1 after training.
4. Evaluate the trained checkpoint on H0 and H1.
5. Compare H0/H1 drift.
6. Estimate column phase correction and write a corrected checkpoint.
7. Evaluate the corrected checkpoint on H1.

Print the command sequence:

```bash
/home/limingfei/miniforge3/envs/speckle/bin/python scripts/plan_measured_tm_workflow.py \
  --config configs/vehicle_dark_intersection_measured_tm_scatter_template.yaml \
  --h0 /data/donn/tm/run_001/H0.npy \
  --h1 /data/donn/tm/run_001/H1.npy \
  --run-dir /data/donn/runs/run_001 \
  --report-dir /data/donn/reports/run_001
```

## Migration Checklist

For a clean lab-server project, copy these first:

- `donn_lab/`
- `models/optical/measured_tm_network.py`
- `scripts/train_measured_tm.py`
- `scripts/evaluate_measured_tm.py`
- `scripts/compare_tm_drift.py`
- `scripts/apply_tm_correction.py`
- `scripts/plan_measured_tm_workflow.py`
- `configs/vehicle_dark_intersection_measured_tm_scatter_template.yaml`

Then copy only the existing training modules that those scripts import:

- `config/`
- `core/`
- `data/`
- `engine/`
- `models/`
- `pipelines/`
- `registry/`

Keep these as external paths, not git content:

- datasets
- measured matrices
- checkpoints
- tensorboard logs
- generated figures and reports

## Five-pass physical inference

The trained five-layer measured-TM network is implemented with five time-multiplexed
passes through the same DMD/scattering-medium/camera path. Start with the inert
hardware preflight:

```powershell
C:\Users\smart\miniconda3\envs\py38\python.exe `
  scripts\run_optical_vehicle_experiment.py --mode hardware --preflight-only
```

The ideal TM simulator and the deliberately armed hardware command are documented in
[`docs/optical_vehicle_experiment.md`](../docs/optical_vehicle_experiment.md).

After remeasuring `tm.npy`, use `scripts/train_and_test_optical_vehicle.py` to
chain fresh training, a simulation accuracy gate, hardware preflight, and an
optional explicitly armed one-sample physical test.

