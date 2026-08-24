from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class DataConfig:
    dataset: str = "synped"
    data_root: str = "/home/limingfei/speckle/donn/datasets/synped"
    label_filter: str = "person"
    multiple_objects: bool = False
    batch_size: int = 32
    max_train_batches: int = 0
    max_test_batches: int = 100
    h_in: int = 128
    w_in: int = 128
    h_out: int = 128
    w_out: int = 128
    vis_samples: int = 6
    input_mode: str = "auto"
    enable_aug: bool = False
    aug_hflip_p: float = 0.5
    aug_color_p: float = 0.8
    aug_blur_p: float = 0.2
    aug_noise_p: float = 0.2
    zoom_crop_mode: str = "none"
    zoom_crop_factor: float = 1.0
    vehicle_channel_mode: str = "rgb"
    vehicle_channel: str = "r"
    vehicle_channel_invert: bool = False
    vehicle_channel_p_low: float = 2.0
    vehicle_channel_p_high: float = 98.0
    vehicle_target_mode: str = "center"
    vehicle_target_column: str = "box_mask"
    # Preserve the historical class-to-focus-coordinate path for old configs.
    # The classification pipeline explicitly switches this to "class".
    mnist_target_mode: str = "coord"


@dataclass
class ModelConfig:
    name: str = "scatter"
    num_layers: int = 1
    seed: int = 42
    activation: str = "abs"
    activation_params: str | dict[str, Any] | None = None
    phase_dropout: float = 0.0
    normalize_input: bool = True
    input_amplitude_normalization: str = "minmax"
    sqrt_amplitude: bool = True
    detector_psf_sigma: float = 0.0
    input_mode: str = "gray"
    phase_init: str = "uniform"
    tmatrix_scale: float = 1.0
    tmatrix_compute_dtype: str = "bf16"
    tmatrix_sparsity: float = 0.98
    tmatrix_path: str | None = None
    tmatrix_shape: str | None = None
    tmatrix_dtype: str = "complex64"
    tmatrix_layout: str = "out_in"
    tmatrix_normalization: str = "none"
    tmatrix_input_h: int = 0
    tmatrix_input_w: int = 0
    tp: bool = False
    return_intensity: bool = True
    normalize_negative: bool = False
    resnet_variant: str = "resnet18"
    resnet_pretrained: bool = False
    tile_h: int = 8
    tile_w: int = 8
    tile_layout_mode: str = "block_average"
    learnable_tile_gains: bool = False
    tile_energy_normalize: bool = True
    learnable_amplitude_bias: bool = False
    cnn_encoder_depth: int = 1
    cnn_encoder_width: int = 8
    cnn_encoder_norm: str = "batchnorm"
    cnn_encoder_activation: str = "relu"
    cnn_encoder_out_activation: str = "softplus"
    mixer_type: str = "spectral"
    mixer_out_activation: str = "softplus"
    mixer_freq_range: float = 0.5
    mixer_mode: str = "phase"
    mixer_init_scale: float = 1e-3
    mixer_kernel_size: int = 9
    mixer_residual: bool = True
    mixer_gate_init: float = 0.0
    mixer_norm: str = "log1p"
    mixer_norm_eps: float = 1e-6
    mixer_domain: str = "intensity"

@dataclass
class OptimConfig:
    lr: float = 1e-3
    epochs: int = 100
    save_every: int = 10


@dataclass
class RuntimeConfig:
    device: str = "cuda:0"
    amp: bool = False
    amp_dtype: str = "bf16"
    resume_ckpt: str | None = None
    resume_from_last: bool = False
    memory_snapshot: bool = False
    memory_snapshot_batch: int = 5


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    ckpt_dir: str = "logs"
    comment: str = ""


@dataclass
class ClearMLConfig:
    enabled: bool = False
    project_name: str = "donn"
    task_name: str | None = None
    tags: list[str] = field(default_factory=list)
    output_uri: str | None = None
    offline: bool = False
    auto_connect_tensorboard: bool = False
    auto_connect_pytorch: bool = False
    report_scalars: bool = True
    report_batch_scalars: bool = False


@dataclass
class LossConfig:
    name: str = "pbr"
    label_smoothing: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    gauss_sigma: float = 1.5
    mix_loss_a: str = "pbr"
    mix_loss_b: str = "mse"
    mix_alpha: float = 0.5


@dataclass
class ClassificationConfig:
    num_classes: int = 10
    grid_rows: int = 2
    grid_cols: int = 5
    roi_h: int = 16
    roi_w: int = 16
    detector_margin: int = 12
    log_energy: bool = True
    efficiency_weight: float = 0.0


@dataclass
class DistillConfig:
    teacher_model: str = "none"
    teacher_ckpt: str | None = None
    task_w: float = 0.0
    kd_pred_w: float = 1.0
    kd_feat_w: float = 1.0
    kd_mode: str = "l2"
    kd_temperature: float = 1.0


@dataclass
class OutputConfig:
    run_name: str | None = None
    schema_version: int = 2


@dataclass
class ExperimentConfig:
    pipeline: str = "base"
    dataset: str = "synped"
    model: str = "scatter"
    data_root: str = "/home/limingfei/speckle/donn/datasets/synped"
    data: DataConfig = field(default_factory=DataConfig)
    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    extras: dict[str, Any] = field(default_factory=dict)


def dataclass_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def _update_dataclass(dc_obj: Any, data: dict[str, Any]) -> None:
    valid = {f.name: f for f in fields(dc_obj)}
    for key, value in data.items():
        if key not in valid:
            continue
        curr = getattr(dc_obj, key)
        if is_dataclass(curr) and isinstance(value, dict):
            _update_dataclass(curr, value)
        else:
            setattr(dc_obj, key, value)


def update_config_from_dict(cfg: ExperimentConfig, payload: dict[str, Any]) -> None:
    _update_dataclass(cfg, payload)
