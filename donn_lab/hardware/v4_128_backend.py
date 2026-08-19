"""Safe adapter for the user's 128 x 128 JUOPT/Spinnaker controller.

The original :mod:`combined_app_v4_128` module was written as an interactive
calibration application.  Importing hardware SDKs and constructing its camera
or DMD classes is therefore deliberately deferred until :meth:`open` is called
with ``arm=True``.  This adapter also owns the complete load/project/capture/
stop/clear transaction so failed frames cannot silently leave uninitialised
pixels in an experiment result.

This module itself is safe to import on machines without PySpin or the JUOPT
DLL.  It intentionally has no dependency on the training YAML configuration.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
from numbers import Integral
import os
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np


PathLike = Union[str, os.PathLike]


class V4128BackendError(RuntimeError):
    """Base class for v4 backend failures."""


class HardwareNotArmedError(V4128BackendError):
    """Raised when code tries to open the devices without explicit consent."""


class HardwareConfigurationError(V4128BackendError):
    """Raised when the controller, encoder, DLL, or device is inconsistent."""


class HardwareTransactionError(V4128BackendError):
    """Raised when a DMD projection/camera capture transaction fails."""


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change cwd, restoring it even when a vendor call fails."""

    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    """Put ``path`` first for a lazy import without permanently polluting it."""

    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        # Remove precisely the entry inserted above.  Imported user modules may
        # have independently changed sys.path, so restoring a full snapshot
        # could accidentally discard their legitimate entries.
        try:
            sys.path.remove(value)
        except ValueError:
            pass


class V4128Backend:
    """Transactional backend around ``combined_app_v4_128``.

    Parameters are stored only by ``__init__``; no SDK is imported and no
    hardware is touched until ``open(arm=True)``.

    Args:
        v4_root: Directory containing the controller helper modules, in
            particular ``dmd_pattern_128.py``.  It also contains the controller
            itself unless ``module_path`` is supplied.
        save_path: Camera output/scratch directory passed to ``CameraHandler``.
        device_name: Exact JUOPT device name.  When omitted, opening succeeds
            only if exactly one DMD is online.
        camera_index: Zero-based Spinnaker camera index.
        module_name: User controller module name, without ``.py``.
        module_path: Optional absolute/relative path to the user's controller
            source.  This permits the controller to live in the experiment
            repository while ``v4_root`` points at its helper modules.  When
            omitted, ``v4_root/module_name.py`` is used.
        dll_parent: Directory relative to which the controller's
            ``JUOPT_DLP ...`` DLL path is resolved.  Defaults to the parent of
            ``v4_root`` (the normal ``Documents/TMCalib`` layout).
        max_retries: Number of complete transaction retries after the initial
            attempt.  A retry restarts camera acquisition to flush stale frames.
    """

    INPUT_HEIGHT = 128
    INPUT_WIDTH = 128
    MAX_SEQUENCE_FRAMES = 1000
    _DLL_RELATIVE_PATH = Path(
        "JUOPT_DLP V4.0.002 20250522 release"
    ) / "4.DLL" / "DLL" / "JUOPT_DLL_V4.dll"

    def __init__(
        self,
        v4_root: PathLike,
        save_path: PathLike,
        device_name: Optional[str] = None,
        camera_index: int = 0,
        module_name: str = "combined_app_v4_128",
        dll_parent: Optional[PathLike] = None,
        max_retries: int = 2,
        module_path: Optional[PathLike] = None,
    ) -> None:
        if isinstance(camera_index, bool) or not isinstance(camera_index, Integral):
            raise TypeError("camera_index must be an integer, not bool or float")
        if isinstance(max_retries, bool) or not isinstance(max_retries, Integral):
            raise TypeError("max_retries must be an integer, not bool or float")
        self.v4_root = Path(v4_root).expanduser().resolve()
        self.save_path = Path(save_path).expanduser().resolve()
        self.device_name = device_name
        self.camera_index = int(camera_index)
        self.module_name = str(module_name)
        self.module_path = (
            Path(module_path).expanduser().resolve()
            if module_path is not None
            else None
        )
        self.dll_parent = (
            Path(dll_parent).expanduser().resolve()
            if dll_parent is not None
            else self.v4_root.parent
        )
        self.max_retries = int(max_retries)

        if self.camera_index < 0:
            raise ValueError("camera_index must be non-negative")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not self.module_name or self.module_name.endswith(".py"):
            raise ValueError("module_name must be a non-empty module name without '.py'")

        self._lock = threading.RLock()
        self._controller_module: Optional[ModuleType] = None
        self._encoder_module: Optional[ModuleType] = None
        self._camera: Optional[Any] = None
        self._dmd: Optional[Any] = None
        self._lut_cache: Optional[Any] = None
        self._opened = False
        self._armed = False
        self._camera_started = False
        self._selected_device: Optional[str] = None
        self._last_frame_id: Optional[int] = None
        self._dmd_faulted = False
        self._transaction_count = 0
        self._retry_count = 0
        self._last_close_errors: List[str] = []
        self._camera_settings: Dict[str, Any] = {}

        self._canvas_height = 768
        self._canvas_width = 1024
        self._active_y = 128
        self._active_x = 256
        self._active_height = 512
        self._active_width = 512

    @property
    def is_open(self) -> bool:
        """Whether both devices are initialized and camera acquisition runs."""

        return self._opened

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return a snapshot suitable for an experiment manifest."""

        controller_file = None
        encoder_file = None
        if self._controller_module is not None:
            controller_file = getattr(self._controller_module, "__file__", None)
        if self._encoder_module is not None:
            encoder_file = getattr(self._encoder_module, "__file__", None)
        return {
            "backend": "V4128Backend",
            "open": bool(self._opened),
            "armed": bool(self._armed),
            "v4_root": str(self.v4_root),
            "controller_module": self.module_name,
            "configured_controller_path": (
                str(self.module_path) if self.module_path is not None else None
            ),
            "controller_file": controller_file,
            "encoder_file": encoder_file,
            "dll_parent": str(self.dll_parent),
            "camera_index": self.camera_index,
            "camera_native_dtype": "uint8",
            "camera_saturation_value": 255,
            "camera_settings": dict(self._camera_settings),
            "requested_device_name": self.device_name,
            "selected_device_name": self._selected_device,
            "input_shape": [self.INPUT_HEIGHT, self.INPUT_WIDTH],
            "dmd_canvas_shape": [self._canvas_height, self._canvas_width],
            "active_region_yxhw": [
                self._active_y,
                self._active_x,
                self._active_height,
                self._active_width,
            ],
            "last_frame_id": self._last_frame_id,
            "dmd_faulted": self._dmd_faulted,
            "successful_transactions": self._transaction_count,
            "transaction_retries": self._retry_count,
            "max_retries": self.max_retries,
            "last_close_errors": list(self._last_close_errors),
        }

    def software_preflight(self) -> Mapping[str, Any]:
        """Import the current controller and exercise encoding without constructing devices.

        This is verified for the supplied controller's current main-guarded
        implementation; executing arbitrary future controller modules cannot
        be proven side-effect-free.
        """

        with self._lock:
            if self._opened or self._armed:
                raise HardwareConfigurationError(
                    "software_preflight must run before hardware is armed"
                )
            self._validate_paths()
            controller, encoder = self._load_user_modules()
            self._controller_module = controller
            self._encoder_module = encoder
            self._load_geometry(encoder)
            probe = np.ones(
                (1, self.INPUT_HEIGHT, self.INPUT_WIDTH), dtype=np.complex64
            )
            pattern = self._encode_fields(probe)
            return {
                "controller_imported": True,
                "encoder_imported": True,
                "controller_file": str(Path(controller.__file__).resolve()),
                "encoder_file": str(Path(encoder.__file__).resolve()),
                "probe_pattern_shape": [int(value) for value in pattern.shape],
                "probe_pattern_dtype": str(pattern.dtype),
                "probe_pattern_values": [int(value) for value in np.unique(pattern)],
                "probe_enabled_pixels": int(np.count_nonzero(pattern)),
                "device_objects_constructed": False,
            }

    def open(self, arm: bool = False) -> "V4128Backend":
        """Initialize camera and DMD only when ``arm`` is literally ``True``."""

        with self._lock:
            if self._opened:
                return self
            if arm is not True:
                raise HardwareNotArmedError(
                    "Hardware remains untouched. Call open(arm=True) to explicitly "
                    "arm the camera and DMD."
                )

            self._validate_paths()
            self._armed = True
            self._selected_device = None
            self._last_close_errors = []
            self._last_frame_id = None
            self._dmd_faulted = False
            self._transaction_count = 0
            self._retry_count = 0
            self._camera_settings = {}
            try:
                controller, encoder = self._load_user_modules()
                self._controller_module = controller
                self._encoder_module = encoder
                self._load_geometry(encoder)

                self.save_path.mkdir(parents=True, exist_ok=True)
                camera_class = self._required_attribute(controller, "CameraHandler")
                dmd_class = self._required_attribute(controller, "DMDController")

                self._camera = camera_class(
                    cam_index=self.camera_index,
                    save_path=str(self.save_path),
                )
                self._configure_camera_checked(controller)
                # The user's DMDController resolves the vendor DLL from cwd.
                # Limit that cwd change to construction and always restore it.
                with _working_directory(self.dll_parent):
                    self._dmd = dmd_class(self._camera)

                if getattr(self._dmd, "DMD", None) is None:
                    raise HardwareConfigurationError(
                        "DMDController could not load the JUOPT DLL"
                    )
                self._validate_controller_geometry()
                self._select_and_initialize_device()

                self._camera.start()
                self._camera_started = True
                self._last_frame_id = None
                self._opened = True
                return self
            except Exception:
                self.close()
                raise

    def project_and_capture_fields(
        self,
        fields: np.ndarray,
        capture_repeats: int = 1,
    ) -> np.ndarray:
        """Encode complex fields, project them, and average repeated captures.

        ``fields`` must be exactly ``complex64[B, 128, 128]``.  Repetition is
        field-major: each encoded field is displayed ``capture_repeats`` times
        consecutively before moving to the next field.
        """

        fields_array = self._validate_fields(fields)
        repeats = self._validate_repeats(capture_repeats)
        total_frames = int(fields_array.shape[0]) * repeats
        self._validate_sequence_length(total_frames)

        with self._lock:
            self._require_open()
            patterns = self._encode_fields(fields_array)
            repeated = np.repeat(patterns, repeats, axis=0)
            frames = self._capture_patterns_with_retries(repeated)
            grouped = frames.reshape(
                fields_array.shape[0],
                repeats,
                self.INPUT_HEIGHT,
                self.INPUT_WIDTH,
            )
            return grouped.mean(axis=1, dtype=np.float32)

    def capture_dark(
        self,
        batch_size: int = 1,
        capture_repeats: int = 1,
    ) -> np.ndarray:
        """Project an all-zero DMD canvas and return grouped dark captures."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
            raise TypeError("batch_size must be an integer, not bool or float")
        count = int(batch_size)
        repeats = self._validate_repeats(capture_repeats)
        if count <= 0:
            raise ValueError("batch_size must be positive")
        self._validate_sequence_length(count * repeats)

        with self._lock:
            self._require_open()
            patterns = np.zeros(
                (count * repeats, self._canvas_height, self._canvas_width),
                dtype=np.uint8,
            )
            frames = self._capture_patterns_with_retries(patterns)
            grouped = frames.reshape(
                count,
                repeats,
                self.INPUT_HEIGHT,
                self.INPUT_WIDTH,
            )
            return grouped.mean(axis=1, dtype=np.float32)

    def blank(self, capture_repeats: int = 1) -> np.ndarray:
        """Display an all-zero canvas and return its averaged camera frame."""

        return self.capture_dark(
            batch_size=1,
            capture_repeats=capture_repeats,
        )[0]

    def close(self) -> None:
        """Blank and release all resources; repeated calls are harmless."""

        with self._lock:
            errors: List[str] = []

            # Attempt one final blank-pattern transaction before ending
            # acquisition. Skip it after any DMD sequence fault: projecting
            # again in an unknown device state would be less safe than going
            # directly to checked stop/free cleanup.
            if (
                self._opened
                and self._camera_started
                and self._dmd is not None
                and not self._dmd_faulted
            ):
                try:
                    blank_pattern = np.zeros(
                        (1, self._canvas_height, self._canvas_width),
                        dtype=np.uint8,
                    )
                    self._capture_patterns_once(
                        blank_pattern,
                        previous_frame_id=None,
                    )
                except Exception as exc:
                    errors.append("final blank failed: {}".format(exc))

            if self._camera is not None and self._camera_started:
                try:
                    if self._camera.stop() is False:
                        errors.append("camera stop returned failure")
                except Exception as exc:
                    errors.append("camera stop failed: {}".format(exc))
                self._camera_started = False

            if self._dmd is not None:
                try:
                    if self._dmd.cleanup() is False:
                        errors.append("DMD cleanup returned failure")
                except Exception as exc:
                    errors.append("DMD cleanup failed: {}".format(exc))

            if self._camera is not None:
                reset_trigger = getattr(self._camera, "reset_trigger", None)
                if callable(reset_trigger):
                    try:
                        if reset_trigger() is False:
                            errors.append("camera trigger reset returned failure")
                    except Exception as exc:
                        errors.append("camera trigger reset failed: {}".format(exc))
                try:
                    if self._camera.cleanup() is False:
                        errors.append("camera cleanup returned failure")
                except Exception as exc:
                    errors.append("camera cleanup failed: {}".format(exc))

            self._dmd = None
            self._camera = None
            self._lut_cache = None
            self._opened = False
            self._armed = False
            self._last_close_errors = errors

            if errors:
                warnings.warn(
                    "V4128Backend closed with cleanup errors: {}".format(
                        "; ".join(errors)
                    ),
                    RuntimeWarning,
                )

    def _configure_camera_checked(self, controller: ModuleType) -> None:
        """Repeat the authoritative setup with checked return values/readback."""

        assert self._camera is not None
        camera = self._camera
        if not hasattr(camera, "convert_to_12bit"):
            raise HardwareConfigurationError(
                "CameraHandler does not expose convert_to_12bit"
            )
        # Polarized8 is the calibrated native precision. A legacy 12-bit
        # conversion would only rescale its 0..255 codes.
        camera.convert_to_12bit = False

        reset_trigger = getattr(camera, "reset_trigger", None)
        if callable(reset_trigger):
            if reset_trigger() is not True:
                raise HardwareConfigurationError(
                    "Camera trigger mode could not be reset before configuration"
                )
        exposure_us = float(getattr(controller, "CAMERA_EXPOSURE_US", 1500.0))
        sdk_camera = getattr(camera, "cam", None)
        if sdk_camera is None:
            raise HardwareConfigurationError("CameraHandler has no initialized SDK camera")

        def read_bool(name: str) -> Optional[bool]:
            node = getattr(sdk_camera, name, None)
            getter = getattr(node, "GetValue", None)
            if callable(getter):
                try:
                    return bool(getter())
                except Exception:
                    pass
            try:
                boolean_node = controller.PySpin.CBooleanPtr(
                    sdk_camera.GetNodeMap().GetNode(name)
                )
                if controller.PySpin.IsReadable(boolean_node):
                    return bool(boolean_node.GetValue())
            except Exception:
                pass
            return None

        checks = (
            ("ISP disable", "configure_ISP", {"isp": False}),
            ("Polarized8 format", "configure_format", {}),
            ("polarization ROI", "configure_roi", {}),
            ("exposure", "configure_exposure", {"exposure_time": exposure_us}),
            ("gamma", "configure_gamma", {"gamma_enable": True, "gamma": 1.0}),
            ("gain", "configure_gain", {"auto_gain": "Off", "gain": 0}),
            ("stream buffer", "configure_buffer_handling", {}),
            ("Line0 trigger", "configure_trigger", {}),
        )
        for label, method_name, kwargs in checks:
            method = getattr(camera, method_name, None)
            if not callable(method):
                raise HardwareConfigurationError(
                    "CameraHandler does not implement {}".format(method_name)
                )
            succeeded = method(**kwargs) is True
            # IspEnable is read-only on some Polarized8 firmware. Accept that
            # only when an independent SDK readback proves ISP is already off.
            if not succeeded and label == "ISP disable" and read_bool("IspEnable") is False:
                succeeded = True
            if not succeeded:
                raise HardwareConfigurationError(
                    "Camera configuration failed at {}".format(label)
                )

        output_roi = (
            int(getattr(camera, "roi_height", -1)),
            int(getattr(camera, "roi_width", -1)),
        )
        if output_roi != (self.INPUT_HEIGHT, self.INPUT_WIDTH):
            raise HardwareConfigurationError(
                "Camera output ROI {} is not the calibrated 128 x 128 grid".format(
                    output_roi
                )
            )
        def read_float(name: str) -> Optional[float]:
            node = getattr(sdk_camera, name, None)
            getter = getattr(node, "GetValue", None)
            return float(getter()) if callable(getter) else None

        actual_exposure = read_float("ExposureTime")
        actual_gain = read_float("Gain")
        actual_gamma = read_float("Gamma")
        actual_isp_enabled = read_bool("IspEnable")
        exposure_tolerance = max(5.0, exposure_us * 0.01)
        if (
            actual_exposure is None
            or abs(actual_exposure - exposure_us) > exposure_tolerance
        ):
            raise HardwareConfigurationError(
                "Camera exposure readback {!r} us differs from requested {:.1f} us".format(
                    actual_exposure, exposure_us
                )
            )
        if actual_gain is None or abs(actual_gain) > 0.05:
            raise HardwareConfigurationError(
                "Camera gain readback {!r} dB is not 0 dB".format(actual_gain)
            )
        if actual_gamma is None or abs(actual_gamma - 1.0) > 0.01:
            raise HardwareConfigurationError(
                "Camera gamma readback {!r} is not 1.0".format(actual_gamma)
            )
        if actual_isp_enabled is not False:
            raise HardwareConfigurationError(
                "Camera ISP readback {!r}; expected disabled".format(
                    actual_isp_enabled
                )
            )

        def read_enum(node_map: Any, name: str) -> Optional[str]:
            try:
                node = controller.PySpin.CEnumerationPtr(node_map.GetNode(name))
                entry = node.GetCurrentEntry()
                return str(entry.GetSymbolic())
            except Exception:
                return None

        node_map = sdk_camera.GetNodeMap()
        stream_map = sdk_camera.GetTLStreamNodeMap()
        enum_expectations = {
            "PixelFormat": (read_enum(node_map, "PixelFormat"), "Polarized8"),
            "TriggerMode": (read_enum(node_map, "TriggerMode"), "On"),
            "TriggerSelector": (read_enum(node_map, "TriggerSelector"), "FrameStart"),
            "TriggerSource": (read_enum(node_map, "TriggerSource"), "Line0"),
            "TriggerOverlap": (read_enum(node_map, "TriggerOverlap"), "ReadOut"),
            "AcquisitionMode": (read_enum(node_map, "AcquisitionMode"), "Continuous"),
            "StreamBufferHandlingMode": (
                read_enum(stream_map, "StreamBufferHandlingMode"),
                "OldestFirst",
            ),
        }
        mismatches = [
            "{}={!r}, expected {!r}".format(name, actual, expected)
            for name, (actual, expected) in enum_expectations.items()
            if actual != expected
        ]
        if mismatches:
            raise HardwareConfigurationError(
                "Camera enum readback mismatch: {}".format("; ".join(mismatches))
            )
        self._camera_settings = {
            "pixel_format": "Polarized8",
            "polarization_quadrant": "I90",
            "output_roi_hw": [self.INPUT_HEIGHT, self.INPUT_WIDTH],
            "exposure_us": actual_exposure,
            "gain_db": actual_gain,
            "gamma": actual_gamma,
            "isp_enabled": actual_isp_enabled,
            "trigger_selector": "FrameStart",
            "trigger_source": "Line0",
            "trigger_overlap": "ReadOut",
            "stream_buffer_mode": "OldestFirst",
            "enum_readback": {
                name: actual for name, (actual, _expected) in enum_expectations.items()
            },
        }

    def __enter__(self) -> "V4128Backend":
        """Enter only after an explicit ``open(arm=True)`` call."""

        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Any],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        self.close()

    def _validate_paths(self) -> None:
        if not self.v4_root.is_dir():
            raise HardwareConfigurationError(
                "v4_root is not a directory: {}".format(self.v4_root)
            )
        module_path = self._controller_path()
        if not module_path.is_file():
            raise HardwareConfigurationError(
                "Controller module was not found: {}".format(module_path)
            )
        encoder_path = self.v4_root / "dmd_pattern_128.py"
        if not encoder_path.is_file():
            raise HardwareConfigurationError(
                "The external 128-grid encoder was not found: {}".format(
                    encoder_path
                )
            )
        if not self.dll_parent.is_dir():
            raise HardwareConfigurationError(
                "dll_parent is not a directory: {}".format(self.dll_parent)
            )
        dll_path = self.dll_parent / self._DLL_RELATIVE_PATH
        if not dll_path.is_file():
            raise HardwareConfigurationError(
                "JUOPT DLL was not found at the controller-relative path: {}".format(
                    dll_path
                )
            )

    def _load_user_modules(self) -> Tuple[ModuleType, ModuleType]:
        """Load the requested source file and its external encoder lazily."""

        module_path = self._controller_path()
        # A unique internal name avoids silently reusing a same-named module
        # previously imported from another checkout.
        internal_name = "_donn_lab_user_v4_128_{}".format(abs(hash(str(module_path))))
        spec = importlib.util.spec_from_file_location(internal_name, str(module_path))
        if spec is None or spec.loader is None:
            raise HardwareConfigurationError(
                "Unable to create an import specification for {}".format(module_path)
            )
        controller = importlib.util.module_from_spec(spec)
        sys.modules[internal_name] = controller
        try:
            with _prepend_sys_path(self.v4_root):
                importlib.invalidate_caches()
                spec.loader.exec_module(controller)
                encoder = importlib.import_module("dmd_pattern_128")
        except Exception:
            sys.modules.pop(internal_name, None)
            raise

        encoder_file_value = getattr(encoder, "__file__", None)
        if encoder_file_value is None:
            raise HardwareConfigurationError("dmd_pattern_128 has no source file")
        encoder_file = Path(encoder_file_value).resolve()
        expected_encoder = (self.v4_root / "dmd_pattern_128.py").resolve()
        if encoder_file != expected_encoder:
            raise HardwareConfigurationError(
                "dmd_pattern_128 was imported from {}, expected {}. Start a clean "
                "process or remove the conflicting module from sys.modules.".format(
                    encoder_file,
                    expected_encoder,
                )
            )
        return controller, encoder

    def _controller_path(self) -> Path:
        if self.module_path is not None:
            return self.module_path
        return self.v4_root / (self.module_name.replace(".", os.sep) + ".py")

    @staticmethod
    def _required_attribute(module: ModuleType, name: str) -> Any:
        value = getattr(module, name, None)
        if value is None:
            raise HardwareConfigurationError(
                "{} does not define {}".format(module.__name__, name)
            )
        return value

    def _load_geometry(self, encoder: ModuleType) -> None:
        names = (
            "DMD_HEIGHT",
            "DMD_WIDTH",
            "INPUT_HEIGHT",
            "INPUT_WIDTH",
            "ACTIVE_Y",
            "ACTIVE_X",
            "ACTIVE_HEIGHT",
            "ACTIVE_WIDTH",
        )
        values: Dict[str, int] = {}
        for name in names:
            raw = self._required_attribute(encoder, name)
            values[name] = int(raw)

        if (values["INPUT_HEIGHT"], values["INPUT_WIDTH"]) != (
            self.INPUT_HEIGHT,
            self.INPUT_WIDTH,
        ):
            raise HardwareConfigurationError(
                "Encoder logical shape is {}, expected (128, 128)".format(
                    (values["INPUT_HEIGHT"], values["INPUT_WIDTH"])
                )
            )

        self._canvas_height = values["DMD_HEIGHT"]
        self._canvas_width = values["DMD_WIDTH"]
        self._active_y = values["ACTIVE_Y"]
        self._active_x = values["ACTIVE_X"]
        self._active_height = values["ACTIVE_HEIGHT"]
        self._active_width = values["ACTIVE_WIDTH"]

        if min(
            self._canvas_height,
            self._canvas_width,
            self._active_height,
            self._active_width,
        ) <= 0:
            raise HardwareConfigurationError("Encoder geometry contains non-positive sizes")
        if self._active_y < 0 or self._active_x < 0:
            raise HardwareConfigurationError("Encoder active-region offsets must be non-negative")
        if self._active_y + self._active_height > self._canvas_height:
            raise HardwareConfigurationError("Encoder active region exceeds DMD height")
        if self._active_x + self._active_width > self._canvas_width:
            raise HardwareConfigurationError("Encoder active region exceeds DMD width")

        self._required_attribute(encoder, "input_field_to_dmd_pattern")
        get_lut = self._required_attribute(encoder, "get_superpixel_lut")
        self._lut_cache = get_lut()

    def _validate_controller_geometry(self) -> None:
        assert self._dmd is not None
        expected = {
            "original_height": self._canvas_height,
            "original_width": self._canvas_width,
            "dmd_height": self.INPUT_HEIGHT,
            "dmd_width": self.INPUT_WIDTH,
            "active_y": self._active_y,
            "active_x": self._active_x,
            "active_height": self._active_height,
            "active_width": self._active_width,
        }
        mismatches: List[str] = []
        for name, expected_value in expected.items():
            actual = getattr(self._dmd, name, None)
            if actual is None or int(actual) != expected_value:
                mismatches.append(
                    "{}={!r} (expected {})".format(name, actual, expected_value)
                )
        if mismatches:
            raise HardwareConfigurationError(
                "DMDController geometry disagrees with dmd_pattern_128: {}".format(
                    ", ".join(mismatches)
                )
            )

    def _select_and_initialize_device(self) -> None:
        assert self._dmd is not None
        devices_raw = self._dmd.get_devices()
        devices = [str(item) for item in devices_raw]
        if self.device_name is None:
            if len(devices) != 1:
                raise HardwareConfigurationError(
                    "Expected exactly one online DMD when device_name is omitted; "
                    "found {}: {}".format(len(devices), devices)
                )
            selected = devices[0]
        else:
            matches = [item for item in devices if item == self.device_name]
            if len(matches) != 1:
                raise HardwareConfigurationError(
                    "Requested DMD {!r} was not uniquely present; online devices: {}".format(
                        self.device_name,
                        devices,
                    )
                )
            selected = matches[0]

        if not bool(self._dmd.initialize_device(selected)):
            raise HardwareConfigurationError(
                "Failed to initialize DMD {!r}".format(selected)
            )
        self._selected_device = selected

    def _validate_fields(self, fields: np.ndarray) -> np.ndarray:
        if not isinstance(fields, np.ndarray):
            raise TypeError("fields must be a numpy.ndarray")
        if fields.dtype != np.dtype(np.complex64):
            raise TypeError(
                "fields dtype must be complex64, got {}".format(fields.dtype)
            )
        expected_tail = (self.INPUT_HEIGHT, self.INPUT_WIDTH)
        if fields.ndim != 3 or tuple(fields.shape[1:]) != expected_tail:
            raise ValueError(
                "fields shape must be (B, 128, 128), got {}".format(fields.shape)
            )
        if fields.shape[0] <= 0:
            raise ValueError("fields batch must be non-empty")
        if not np.all(np.isfinite(fields)):
            raise ValueError("fields contain NaN or infinite values")
        return np.ascontiguousarray(fields)

    @staticmethod
    def _validate_repeats(capture_repeats: int) -> int:
        if isinstance(capture_repeats, bool) or not isinstance(
            capture_repeats,
            Integral,
        ):
            raise TypeError(
                "capture_repeats must be an integer, not bool or float"
            )
        repeats = int(capture_repeats)
        if repeats <= 0:
            raise ValueError("capture_repeats must be a positive integer")
        return repeats

    def _validate_sequence_length(self, count: int) -> None:
        if count <= 0 or count > self.MAX_SEQUENCE_FRAMES:
            raise ValueError(
                "A hardware transaction must contain 1..{} frames; got {}".format(
                    self.MAX_SEQUENCE_FRAMES,
                    count,
                )
            )

    def _encode_fields(self, fields: np.ndarray) -> np.ndarray:
        assert self._encoder_module is not None
        encoder = self._required_attribute(
            self._encoder_module,
            "input_field_to_dmd_pattern",
        )
        px = int(
            getattr(
                self._encoder_module,
                "HOLOGRAM_SUPERPIXEL_SIZE",
                4,
            )
        )
        patterns = np.empty(
            (fields.shape[0], self._canvas_height, self._canvas_width),
            dtype=np.uint8,
        )
        for index, field in enumerate(fields):
            # The external encoder normalizes by maximum amplitude and rejects
            # a zero field.  Physically, a zero field is exactly a blank DMD.
            if not np.any(field):
                patterns[index].fill(0)
                continue
            encoded = encoder(
                field,
                px=px,
                ds_method="mean",
                lut_cache=self._lut_cache,
            )
            patterns[index] = self._validate_single_pattern(encoded, index)
        self._validate_patterns(patterns)
        return patterns

    def _validate_single_pattern(self, pattern: Any, index: int) -> np.ndarray:
        array = np.asarray(pattern)
        expected = (self._canvas_height, self._canvas_width)
        if array.shape != expected:
            raise HardwareConfigurationError(
                "Encoder pattern {} has shape {}, expected {}".format(
                    index,
                    array.shape,
                    expected,
                )
            )
        if array.dtype != np.dtype(np.uint8):
            raise HardwareConfigurationError(
                "Encoder pattern {} has dtype {}, expected uint8".format(
                    index,
                    array.dtype,
                )
            )
        return np.ascontiguousarray(array)

    def _validate_patterns(self, patterns: np.ndarray) -> None:
        expected_tail = (self._canvas_height, self._canvas_width)
        if not isinstance(patterns, np.ndarray):
            raise TypeError("patterns must be a numpy.ndarray")
        if patterns.dtype != np.dtype(np.uint8):
            raise TypeError("patterns dtype must be uint8")
        if patterns.ndim != 3 or tuple(patterns.shape[1:]) != expected_tail:
            raise ValueError(
                "patterns shape must be (B, {}, {}), got {}".format(
                    self._canvas_height,
                    self._canvas_width,
                    patterns.shape,
                )
            )
        self._validate_sequence_length(int(patterns.shape[0]))

        # The vendor SDK is configured with sig_bit=1.  Requiring canonical
        # 0/255 bytes prevents ambiguous low-valued "on" pixels.
        if np.any((patterns != 0) & (patterns != 255)):
            bad_values = np.unique(patterns[(patterns != 0) & (patterns != 255)])
            raise HardwareConfigurationError(
                "DMD patterns must be binary uint8 values {0, 255}; found {}".format(
                    bad_values[:8].tolist()
                )
            )

        y0 = self._active_y
        y1 = y0 + self._active_height
        x0 = self._active_x
        x1 = x0 + self._active_width
        margins_on = (
            np.any(patterns[:, :y0, :])
            or np.any(patterns[:, y1:, :])
            or np.any(patterns[:, y0:y1, :x0])
            or np.any(patterns[:, y0:y1, x1:])
        )
        if margins_on:
            raise HardwareConfigurationError(
                "DMD pattern contains enabled pixels outside the configured active region"
            )

    def _capture_patterns_with_retries(self, patterns: np.ndarray) -> np.ndarray:
        self._validate_patterns(patterns)
        failures: List[str] = []
        for attempt in range(self.max_retries + 1):
            # Start every transaction with a fresh acquisition session so a
            # trigger left around juoptStop/clear cannot shift patterns by one
            # while still presenting consecutive FrameIDs.
            self._restart_camera()
            try:
                frames, frame_ids = self._capture_patterns_once(
                    patterns,
                    previous_frame_id=None,
                )
                self._last_frame_id = frame_ids[-1]
                self._transaction_count += 1
                return frames
            except Exception as exc:
                failures.append("attempt {}: {}".format(attempt + 1, exc))
                if isinstance(exc, HardwareConfigurationError):
                    # DMD stop/clear or configuration failures leave sequence
                    # state unknown and must never be retried in-place.
                    raise
                if attempt >= self.max_retries:
                    break
                self._retry_count += 1

        raise HardwareTransactionError(
            "Projection/capture failed after {} attempt(s): {}".format(
                self.max_retries + 1,
                " | ".join(failures),
            )
        )

    def _capture_patterns_once(
        self,
        patterns: np.ndarray,
        previous_frame_id: Optional[int],
    ) -> Tuple[np.ndarray, List[int]]:
        """Run one indivisible load/project/capture/stop/clear transaction."""

        self._require_open_or_opening()
        self._validate_patterns(patterns)
        assert self._dmd is not None
        assert self._camera is not None

        if not bool(self._dmd.load_pattern(np.ascontiguousarray(patterns))):
            cleanup_errors: List[str] = []
            try:
                stop_result = self._dmd.DMD.juoptStop(self._dmd.dev_id)
                if stop_result not in (0, None):
                    cleanup_errors.append("juoptStop code {}".format(stop_result))
            except Exception as exc:
                cleanup_errors.append("juoptStop exception {}".format(exc))
            try:
                if not bool(self._dmd.clear_sequence(0)):
                    cleanup_errors.append("clear_sequence returned failure")
            except Exception as exc:
                cleanup_errors.append("clear_sequence exception {}".format(exc))
            if cleanup_errors:
                self._dmd_faulted = True
                raise HardwareConfigurationError(
                    "DMD load_pattern failed and sequence recovery failed: {}".format(
                        "; ".join(cleanup_errors)
                    )
                )
            raise HardwareTransactionError("DMD load_pattern returned failure")

        frames: List[np.ndarray] = []
        frame_ids: List[int] = []
        primary_error: Optional[Exception] = None
        cleanup_errors: List[str] = []
        try:
            result = self._dmd.DMD.juoptProjection(self._dmd.dev_id, 0, 0)
            if result not in (0, None):
                raise HardwareTransactionError(
                    "juoptProjection failed with code {}".format(result)
                )

            for frame_index in range(patterns.shape[0]):
                captured, _wait_time, _process_time = self._camera.run()
                if captured is None:
                    raise HardwareTransactionError(
                        "camera returned no image for transaction frame {}".format(
                            frame_index
                        )
                    )
                image = np.asarray(captured)
                expected_shape = (self.INPUT_HEIGHT, self.INPUT_WIDTH)
                if image.shape != expected_shape:
                    raise HardwareTransactionError(
                        "camera frame {} has shape {}, expected {}".format(
                            frame_index,
                            image.shape,
                            expected_shape,
                        )
                    )
                if image.dtype != np.dtype(np.uint8):
                    raise HardwareTransactionError(
                        "camera frame {} has dtype {}, expected native uint8".format(
                            frame_index,
                            image.dtype,
                        )
                    )
                if not np.all(np.isfinite(image)):
                    raise HardwareTransactionError(
                        "camera frame {} contains NaN or infinity".format(frame_index)
                    )

                raw_frame_id = getattr(self._camera, "last_frame_id", None)
                if raw_frame_id is None:
                    raise HardwareTransactionError(
                        "camera did not provide FrameID for transaction frame {}".format(
                            frame_index
                        )
                    )
                frame_id = int(raw_frame_id)
                frames.append(np.asarray(image, dtype=np.float32))
                frame_ids.append(frame_id)
        except Exception as exc:
            primary_error = exc
        finally:
            try:
                stop_result = self._dmd.DMD.juoptStop(self._dmd.dev_id)
                if stop_result not in (0, None):
                    cleanup_errors.append(
                        "juoptStop code {}".format(stop_result)
                    )
            except Exception as exc:
                cleanup_errors.append("juoptStop exception {}".format(exc))
            try:
                if not bool(self._dmd.clear_sequence(0)):
                    cleanup_errors.append("clear_sequence returned failure")
            except Exception as exc:
                cleanup_errors.append("clear_sequence exception {}".format(exc))

        if primary_error is not None:
            if cleanup_errors:
                self._dmd_faulted = True
                raise HardwareConfigurationError(
                    "{}; cleanup also failed: {}".format(
                        primary_error,
                        "; ".join(cleanup_errors),
                    )
                )
            if isinstance(primary_error, HardwareTransactionError):
                raise primary_error
            raise HardwareTransactionError(str(primary_error))
        if cleanup_errors:
            self._dmd_faulted = True
            raise HardwareConfigurationError(
                "Transaction cleanup failed: {}".format("; ".join(cleanup_errors))
            )

        self._validate_frame_ids(frame_ids, previous_frame_id)
        return np.stack(frames, axis=0).astype(np.float32, copy=False), frame_ids

    @staticmethod
    def _validate_frame_ids(
        frame_ids: Sequence[int],
        previous_frame_id: Optional[int],
    ) -> None:
        if not frame_ids:
            raise HardwareTransactionError("transaction returned no camera FrameIDs")
        if previous_frame_id is not None and frame_ids[0] != previous_frame_id + 1:
            raise HardwareTransactionError(
                "camera FrameID discontinuity across transactions: {} -> {}".format(
                    previous_frame_id,
                    frame_ids[0],
                )
            )
        for index in range(1, len(frame_ids)):
            if frame_ids[index] != frame_ids[index - 1] + 1:
                raise HardwareTransactionError(
                    "camera FrameID discontinuity inside transaction at frame {}: "
                    "{} -> {}".format(
                        index,
                        frame_ids[index - 1],
                        frame_ids[index],
                    )
                )

    def _restart_camera(self) -> None:
        assert self._camera is not None
        try:
            if self._camera_started:
                if self._camera.stop() is False:
                    raise HardwareTransactionError(
                        "camera acquisition stop returned failure"
                    )
        finally:
            self._camera_started = False
        try:
            self._camera.start()
        except Exception as exc:
            raise HardwareTransactionError(
                "failed to restart camera acquisition: {}".format(exc)
            )
        self._camera_started = True
        self._last_frame_id = None

    def _require_open(self) -> None:
        if not self._opened:
            raise HardwareConfigurationError(
                "Backend is closed; call open(arm=True) before capture"
            )

    def _require_open_or_opening(self) -> None:
        # ``close`` captures a final blank while _opened is still true.  During
        # normal transactions all four conditions are also required.
        if (
            not self._opened
            or not self._camera_started
            or self._camera is None
            or self._dmd is None
        ):
            raise HardwareConfigurationError(
                "Camera/DMD transaction requested before successful open"
            )


__all__ = [
    "HardwareConfigurationError",
    "HardwareNotArmedError",
    "HardwareTransactionError",
    "V4128Backend",
    "V4128BackendError",
]
