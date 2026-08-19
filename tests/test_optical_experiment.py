from __future__ import annotations

import json
import gc
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from donn_lab.experiment.artifacts import RunArtifactWriter
from donn_lab.experiment.optical_inference import (
    CheckpointPhases,
    OpticalInferenceRunner,
    OpticalInferenceSettings,
    load_checkpoint_phases,
)
from donn_lab.hardware.v4_128_backend import (
    HardwareConfigurationError,
    HardwareNotArmedError,
    V4128Backend,
)
from donn_lab.optics.detector_psf import GaussianIntensityPSF


def _measured_tm_network_class():
    source = Path(__file__).resolve().parents[1] / "models/optical/measured_tm_network.py"
    spec = importlib.util.spec_from_file_location("_measured_tm_network_test", str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load measured_tm_network.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MeasuredTMScatterNetwork


class NumpyTMBackend:
    def __init__(
        self,
        matrix: np.ndarray,
        height: int,
        width: int,
        detector_psf_sigma: float = 0.0,
    ) -> None:
        self.matrix = np.asarray(matrix, dtype=np.complex64)
        self.input_height = height
        self.input_width = width
        self.height = height
        self.width = width
        self.opened = False
        self.detector_psf = GaussianIntensityPSF(detector_psf_sigma)

    @property
    def metadata(self):
        return {"backend": "numpy-test"}

    def open(self):
        self.opened = True

    def close(self):
        self.opened = False

    def capture_dark(self, batch_size=1, capture_repeats=1):
        return np.zeros((batch_size, self.height, self.width), dtype=np.float32)

    def project_and_capture_fields(self, fields, capture_repeats=1):
        vectors = np.asarray(fields, dtype=np.complex64).reshape(fields.shape[0], -1)
        detector = (self.matrix @ vectors.T).T
        intensity = np.square(np.abs(detector)).reshape(
            -1, 1, self.height, self.width
        ).astype(np.float32)
        with torch.no_grad():
            blurred = self.detector_psf(torch.from_numpy(intensity))
        return blurred[:, 0].numpy()


class OpticalInferenceParityTest(unittest.TestCase):
    @unittest.skipIf(sys.version_info < (3, 10), "training model dependencies require Python 3.10+")
    def test_closed_loop_matches_training_model(self) -> None:
        MeasuredTMScatterNetwork = _measured_tm_network_class()
        rng = np.random.default_rng(7)
        height = width = 4
        matrix = (
            rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        ).astype(np.complex64)
        images = rng.random((3, 1, height, width), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            matrix_path = Path(temporary) / "tm.npy"
            np.save(str(matrix_path), matrix)
            model = MeasuredTMScatterNetwork(
                (height, width),
                (height, width),
                tmatrix_path=matrix_path,
                normalize_input=True,
                sqrt_amplitude=True,
                detector_psf_sigma=0.6,
                phase_init="uniform",
                num_layers=3,
                seed=11,
                device=torch.device("cpu"),
                activation="abs",
            )
            model.eval()
            phases = np.stack(
                [parameter.detach().cpu().numpy()[0, 0] for parameter in model.phases],
                axis=0,
            ).astype(np.float32)
            checkpoint = CheckpointPhases(
                phases=phases,
                checkpoint_path="synthetic",
                input_hw=(height, width),
                mode_hw=(height, width),
                output_hw=(height, width),
                normalize_input=True,
                sqrt_amplitude=True,
                detector_psf_sigma=0.6,
                model_name="measured_tm_scatter",
            )
            backend = NumpyTMBackend(matrix, height, width, detector_psf_sigma=0.6)
            runner = OpticalInferenceRunner(
                checkpoint,
                backend,
                OpticalInferenceSettings(auto_capture_dark=False),
            )
            result = runner.infer(images, return_traces=True)
            runner.close()
            with torch.no_grad():
                expected = model(torch.from_numpy(images)).cpu().numpy()
            del model
            gc.collect()

        np.testing.assert_allclose(result.final_intensity, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(len(result.traces), 3)
        # The second layer's pre-normalized input is the first camera intensity,
        # not its square root. Parity above would fail if this invariant changed.

    @unittest.skipIf(sys.version_info < (3, 10), "training model dependencies require Python 3.10+")
    def test_png_amplitude_max_normalization_matches_training_model(self) -> None:
        MeasuredTMScatterNetwork = _measured_tm_network_class()
        rng = np.random.default_rng(19)
        height = width = 3
        matrix = (
            rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
        ).astype(np.complex64)
        images = np.asarray(
            [
                [[[0.10, 0.15, 0.20], [0.25, 0.40, 0.60], [0.80, 0.90, 1.00]]],
                [[[0.05, 0.08, 0.12], [0.20, 0.30, 0.45], [0.55, 0.70, 0.85]]],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temporary:
            matrix_path = Path(temporary) / "tm.npy"
            np.save(str(matrix_path), matrix)
            model = MeasuredTMScatterNetwork(
                (height, width),
                (height, width),
                tmatrix_path=matrix_path,
                normalize_input=True,
                input_amplitude_normalization="max",
                sqrt_amplitude=False,
                phase_init="uniform",
                num_layers=2,
                seed=23,
                device=torch.device("cpu"),
                activation="abs",
            )
            model.eval()
            phases = np.stack(
                [parameter.detach().cpu().numpy()[0, 0] for parameter in model.phases],
                axis=0,
            ).astype(np.float32)
            checkpoint = CheckpointPhases(
                phases=phases,
                checkpoint_path="synthetic-amplitude",
                input_hw=(height, width),
                mode_hw=(height, width),
                output_hw=(height, width),
                normalize_input=True,
                sqrt_amplitude=False,
                input_amplitude_normalization="max",
                model_name="measured_tm_scatter",
            )
            runner = OpticalInferenceRunner(
                checkpoint,
                NumpyTMBackend(matrix, height, width),
                OpticalInferenceSettings(auto_capture_dark=False),
            )
            result = runner.infer(images, return_traces=True)
            runner.close()
            with torch.no_grad():
                expected = model(torch.from_numpy(images)).cpu().numpy()
            del model
            gc.collect()

        np.testing.assert_allclose(result.final_intensity, expected, rtol=2e-5, atol=2e-5)
        wanted_first_amplitude = images[:, 0] / images[:, 0].max(
            axis=(1, 2), keepdims=True
        )
        np.testing.assert_allclose(
            result.traces[0].projected_amplitude,
            wanted_first_amplitude,
            rtol=2e-6,
            atol=2e-6,
        )
        self.assertGreater(float(result.traces[0].projected_amplitude.min()), 0.0)

    def test_checkpoint_keeps_raw_periodic_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            raw_phase = torch.tensor([[[[-1.0, 8.0], [15.0, -7.0]]]], dtype=torch.float32)
            torch.save(
                {
                    "model_state_dict": {"phases.0": raw_phase},
                    "config_dump": {
                        "model": "measured_tm_scatter",
                        "model_cfg": {
                            "name": "measured_tm_scatter",
                            "num_layers": 1,
                            "activation": "abs",
                            "normalize_input": True,
                            "input_amplitude_normalization": "max",
                            "sqrt_amplitude": False,
                            "detector_psf_sigma": 0.6,
                        },
                        "data": {"h_in": 2, "w_in": 2, "h_out": 2, "w_out": 2},
                    },
                },
                str(path),
            )
            loaded = load_checkpoint_phases(path, expected_num_layers=1, expected_hw=(2, 2))
        np.testing.assert_array_equal(loaded.phases[0], raw_phase.numpy()[0, 0])
        self.assertEqual(loaded.input_amplitude_normalization, "max")
        self.assertFalse(loaded.sqrt_amplitude)
        self.assertAlmostEqual(loaded.detector_psf_sigma, 0.6)


class ExperimentSafetyTest(unittest.TestCase):
    def test_v4_backend_cannot_open_without_explicit_arm(self) -> None:
        backend = V4128Backend(v4_root="missing", save_path="unused")
        with self.assertRaises(HardwareNotArmedError):
            backend.open()
        self.assertFalse(backend.is_open)

    def test_v4_sequence_recovery_failure_disables_final_projection(self) -> None:
        class FakeSDK:
            @staticmethod
            def juoptStop(_device_id):
                return 0

        class FakeDMD:
            def __init__(self):
                self.DMD = FakeSDK()
                self.dev_id = 1
                self.load_calls = 0

            def load_pattern(self, _patterns):
                self.load_calls += 1
                return False

            @staticmethod
            def clear_sequence(_sequence_id):
                return False

            @staticmethod
            def cleanup():
                return True

        class FakeCamera:
            @staticmethod
            def stop():
                return True

            @staticmethod
            def reset_trigger():
                return True

            @staticmethod
            def cleanup():
                return True

        backend = V4128Backend(v4_root="unused", save_path="unused")
        backend._dmd = FakeDMD()
        backend._camera = FakeCamera()
        backend._opened = True
        backend._armed = True
        backend._camera_started = True
        patterns = np.zeros((1, 768, 1024), dtype=np.uint8)
        with self.assertRaises(HardwareConfigurationError):
            backend._capture_patterns_once(patterns, previous_frame_id=None)
        self.assertTrue(backend._dmd_faulted)
        dmd = backend._dmd
        backend.close()
        self.assertEqual(dmd.load_calls, 1)

    def test_artifact_resume_requires_identical_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            writer = RunArtifactWriter(root, {"fingerprint": {"sha256": "a"}})
            writer.prepare()
            writer.write_sample(
                sample_id="sample-1",
                input_chw=np.zeros((1, 2, 2), dtype=np.float32),
                final_intensity=np.ones((2, 2), dtype=np.float32),
                prediction={"pixel_distance": 0.0},
            )
            self.assertTrue(writer.sample_done("sample-1"))
            # Simulate a crash after the atomic sample rename but before the
            # manifest append; resume must reconstruct the row from prediction.
            (root / "manifest.csv").unlink()
            resumed = RunArtifactWriter(
                root,
                {"fingerprint": {"sha256": "a"}},
                resume=True,
            )
            resumed.prepare()
            self.assertIn("sample-1", (root / "manifest.csv").read_text(encoding="utf-8"))
            mismatch = RunArtifactWriter(
                root,
                {"fingerprint": {"sha256": "b"}},
                resume=True,
            )
            with self.assertRaises(RuntimeError):
                mismatch.prepare()
            self.assertTrue(json.loads((root / "samples/sample-1/DONE.json").read_text())["sample_id"])


if __name__ == "__main__":
    unittest.main()
