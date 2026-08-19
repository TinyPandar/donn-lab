from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

from config.schema import ExperimentConfig
from registry import create_model, register_model

from .cnn_backbones import ResNetHeatmap, SimpleCNN, SimpleDistCNN
from .matmul_cnn import MatMulCNN


@dataclass
class DistContext:
    is_dist: bool
    local_rank: int
    world_size: int


def _activation_params(cfg: ExperimentConfig) -> dict[str, Any]:
    params = cfg.model_cfg.activation_params
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    if isinstance(params, str):
        try:
            return json.loads(params)
        except json.JSONDecodeError:
            print(f"Warning: failed to parse activation_params JSON: {params}")
            return {}
    return {}


def _input_channels(cfg: ExperimentConfig) -> int:
    mode = str(cfg.data.input_mode).lower()
    return 3 if mode in {"rgb", "bgr", "none", "off"} else 1


def _build_scatter(cfg: ExperimentConfig, device: torch.device, dist_ctx: DistContext):
    from models.optical.scatter_neural_network import ScatterNeuralNetwork

    return ScatterNeuralNetwork(
        input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)),
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
        num_layers=int(cfg.model_cfg.num_layers),
        seed=int(cfg.model_cfg.seed),
        tmatrix_compute_dtype=(
            None
            if str(cfg.model_cfg.tmatrix_compute_dtype) == "fp32"
            else (torch.bfloat16 if str(cfg.model_cfg.tmatrix_compute_dtype) == "bf16" else torch.float16)
        ),
        tp_enabled=bool(cfg.model_cfg.tp and dist_ctx.world_size > 1),
        tp_rank=(dist_ctx.local_rank if dist_ctx.world_size > 1 else 0),
        tp_world_size=(dist_ctx.world_size if dist_ctx.world_size > 1 else 1),
        activation=str(cfg.model_cfg.activation),
        activation_params=_activation_params(cfg),
        phase_dropout=float(cfg.model_cfg.phase_dropout),
        normalize_input=bool(cfg.model_cfg.normalize_input),
        sqrt_amplitude=bool(cfg.model_cfg.sqrt_amplitude),
        phase_init=str(cfg.model_cfg.phase_init),
        tmatrix_scale=float(cfg.model_cfg.tmatrix_scale),
    ).to(device)


def _build_measured_tm_scatter(cfg: ExperimentConfig, device: torch.device, dist_ctx: DistContext):
    _ = dist_ctx
    from models.optical.measured_tm_network import MeasuredTMScatterNetwork

    input_modes_hw = None
    if int(cfg.model_cfg.tmatrix_input_h) > 0 or int(cfg.model_cfg.tmatrix_input_w) > 0:
        if int(cfg.model_cfg.tmatrix_input_h) <= 0 or int(cfg.model_cfg.tmatrix_input_w) <= 0:
            raise ValueError("tmatrix_input_h and tmatrix_input_w must be set together")
        input_modes_hw = (int(cfg.model_cfg.tmatrix_input_h), int(cfg.model_cfg.tmatrix_input_w))

    return MeasuredTMScatterNetwork(
        input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)),
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
        tmatrix_path=cfg.model_cfg.tmatrix_path,
        tmatrix_shape=cfg.model_cfg.tmatrix_shape,
        tmatrix_dtype=str(cfg.model_cfg.tmatrix_dtype),
        tmatrix_layout=str(cfg.model_cfg.tmatrix_layout),
        tmatrix_normalization=str(cfg.model_cfg.tmatrix_normalization),
        input_modes_hw=input_modes_hw,
        num_layers=int(cfg.model_cfg.num_layers),
        seed=int(cfg.model_cfg.seed),
        activation=str(cfg.model_cfg.activation),
        activation_params=_activation_params(cfg),
        phase_dropout=float(cfg.model_cfg.phase_dropout),
        normalize_input=bool(cfg.model_cfg.normalize_input),
        input_amplitude_normalization=str(
            cfg.model_cfg.input_amplitude_normalization
        ),
        sqrt_amplitude=bool(cfg.model_cfg.sqrt_amplitude),
        detector_psf_sigma=float(cfg.model_cfg.detector_psf_sigma),
        phase_init=str(cfg.model_cfg.phase_init),
        device=device,
    ).to(device)


def _build_scatter_mixer(cfg: ExperimentConfig, device: torch.device, dist_ctx: DistContext):
    from models.optical.scatter_mixer_network import ScatterMixerNetwork

    return ScatterMixerNetwork(
        input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)),
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
        num_layers=int(cfg.model_cfg.num_layers),
        return_intensity=bool(cfg.model_cfg.return_intensity),
        tmatrix_compute_dtype=str(cfg.model_cfg.tmatrix_compute_dtype),
        tmatrix_sparsity=float(cfg.model_cfg.tmatrix_sparsity),
        tp_enabled=bool(cfg.model_cfg.tp and dist_ctx.world_size > 1),
        tp_rank=(dist_ctx.local_rank if dist_ctx.world_size > 1 else 0),
        tp_world_size=(dist_ctx.world_size if dist_ctx.world_size > 1 else 1),
        activation=str(cfg.model_cfg.activation),
        activation_params=_activation_params(cfg),
        normalize_negative=bool(cfg.model_cfg.normalize_negative),
        phase_dropout=float(cfg.model_cfg.phase_dropout),
        mixer_type=str(cfg.model_cfg.mixer_type),
        mixer_out_activation=str(cfg.model_cfg.mixer_out_activation),
        mixer_freq_range=float(cfg.model_cfg.mixer_freq_range),
        mixer_mode=str(cfg.model_cfg.mixer_mode),
        mixer_init_scale=float(cfg.model_cfg.mixer_init_scale),
        mixer_kernel_size=int(cfg.model_cfg.mixer_kernel_size),
        mixer_residual=bool(cfg.model_cfg.mixer_residual),
        mixer_gate_init=float(cfg.model_cfg.mixer_gate_init),
        mixer_norm=str(cfg.model_cfg.mixer_norm),
        mixer_norm_eps=float(cfg.model_cfg.mixer_norm_eps),
        mixer_domain=str(cfg.model_cfg.mixer_domain),
    ).to(device)


def _build_scatter_tile(cfg: ExperimentConfig, device: torch.device, dist_ctx: DistContext):
    from models.optical.scatter_tile_neural_network import ScatterTileNeuralNetwork

    return ScatterTileNeuralNetwork(
        input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)),
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
        normalize_input=bool(cfg.model_cfg.normalize_input),
        sqrt_amplitude=bool(cfg.model_cfg.sqrt_amplitude),
        phase_init=str(cfg.model_cfg.phase_init),
        tmatrix_scale=float(cfg.model_cfg.tmatrix_scale),
        num_layers=int(cfg.model_cfg.num_layers),
        seed=int(cfg.model_cfg.seed),
        device=device,
        tp_enabled=bool(cfg.model_cfg.tp and dist_ctx.world_size > 1),
        tp_rank=(dist_ctx.local_rank if dist_ctx.world_size > 1 else 0),
        tp_world_size=(dist_ctx.world_size if dist_ctx.world_size > 1 else 1),
        activation=str(cfg.model_cfg.activation),
        phase_dropout=float(cfg.model_cfg.phase_dropout),
        tile_hw=(int(cfg.model_cfg.tile_h), int(cfg.model_cfg.tile_w)),
        layout_mode=str(cfg.model_cfg.tile_layout_mode),
        learnable_tile_gains=bool(cfg.model_cfg.learnable_tile_gains),
        energy_normalize=bool(cfg.model_cfg.tile_energy_normalize),
        learnable_amplitude_bias=bool(cfg.model_cfg.learnable_amplitude_bias),
    ).to(device)


def _build_simple_cnn(cfg: ExperimentConfig, device: torch.device, _: DistContext):
    return SimpleCNN(input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)), output_hw=(int(cfg.data.h_out), int(cfg.data.w_out))).to(device)


def _build_simple_dist_cnn(cfg: ExperimentConfig, device: torch.device, _: DistContext):
    return SimpleDistCNN(input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)), output_hw=(int(cfg.data.h_out), int(cfg.data.w_out))).to(device)


def _build_resnet(cfg: ExperimentConfig, device: torch.device, _: DistContext):
    return ResNetHeatmap(
        input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)),
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
        variant=str(cfg.model_cfg.resnet_variant),
        pretrained=bool(cfg.model_cfg.resnet_pretrained),
    ).to(device)


def _build_matmul_cnn(cfg: ExperimentConfig, device: torch.device, _: DistContext):
    return MatMulCNN(
        input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)),
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
    ).to(device)


def _build_cnn_scatter(cfg: ExperimentConfig, device: torch.device, dist_ctx: DistContext):
    from .hybrid_cnn_scatter import CnnScatterNetwork

    return CnnScatterNetwork(
        input_hw=(int(cfg.data.h_in), int(cfg.data.w_in)),
        output_hw=(int(cfg.data.h_out), int(cfg.data.w_out)),
        cnn_encoder_depth=int(cfg.model_cfg.cnn_encoder_depth),
        cnn_encoder_width=int(cfg.model_cfg.cnn_encoder_width),
        in_channels=_input_channels(cfg),
        cnn_encoder_norm=str(cfg.model_cfg.cnn_encoder_norm),
        cnn_encoder_activation=str(cfg.model_cfg.cnn_encoder_activation),
        cnn_encoder_out_activation=str(cfg.model_cfg.cnn_encoder_out_activation),
        num_layers=int(cfg.model_cfg.num_layers),
        seed=int(cfg.model_cfg.seed),
        tmatrix_compute_dtype=(
            None
            if str(cfg.model_cfg.tmatrix_compute_dtype) == "fp32"
            else (torch.bfloat16 if str(cfg.model_cfg.tmatrix_compute_dtype) == "bf16" else torch.float16)
        ),
        tp_enabled=bool(cfg.model_cfg.tp and dist_ctx.world_size > 1),
        tp_rank=(dist_ctx.local_rank if dist_ctx.world_size > 1 else 0),
        tp_world_size=(dist_ctx.world_size if dist_ctx.world_size > 1 else 1),
        activation=str(cfg.model_cfg.activation),
        activation_params=_activation_params(cfg),
        phase_dropout=float(cfg.model_cfg.phase_dropout),
        normalize_input=bool(cfg.model_cfg.normalize_input),
        sqrt_amplitude=bool(cfg.model_cfg.sqrt_amplitude),
        phase_init=str(cfg.model_cfg.phase_init),
        tmatrix_scale=float(cfg.model_cfg.tmatrix_scale),
    ).to(device)


def register_builtin_models() -> None:
    registrations = {
        "scatter": _build_scatter,
        "measured_tm_scatter": _build_measured_tm_scatter,
        "scatter_mixer": _build_scatter_mixer,
        "scatter_tile": _build_scatter_tile,
        "stn": _build_scatter_tile,
        "cnn_scatter": _build_cnn_scatter,
        "simple_cnn": _build_simple_cnn,
        "simple_dist_cnn": _build_simple_dist_cnn,
        "resnet": _build_resnet,
        "matmul_cnn": _build_matmul_cnn,
    }
    for name, fn in registrations.items():
        try:
            register_model(name, fn)
        except ValueError:
            pass


def create_registered_model(name: str, cfg: ExperimentConfig, device: torch.device, dist_ctx: DistContext):
    return create_model(name, cfg, device, dist_ctx)
