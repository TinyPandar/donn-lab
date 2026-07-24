from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import torch

from config.schema import ExperimentConfig


class CheckpointIO:
    SCHEMA_VERSION = 2

    @classmethod
    def _latest_ckpt(cls, ckpt_dir: str) -> str | None:
        if not os.path.isdir(ckpt_dir):
            return None
        candidates = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        if not candidates:
            return None
        candidates.sort(key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x else -1)
        return os.path.join(ckpt_dir, candidates[-1])

    @classmethod
    def resolve_resume_path(cls, cfg: ExperimentConfig) -> str | None:
        if cfg.runtime.resume_ckpt:
            return cfg.runtime.resume_ckpt
        if cfg.runtime.resume_from_last:
            return cls._latest_ckpt(cfg.logging.ckpt_dir)
        return None

    @classmethod
    def save(
        cls,
        *,
        path: str,
        epoch: int,
        global_step: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        train_loss_avg: float | None,
        test_loss_avg: float | None,
        cfg: ExperimentConfig,
    ) -> None:
        ckpt = {
            "schema_version": int(cls.SCHEMA_VERSION),
            "pipeline": cfg.pipeline,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "train_loss_avg": train_loss_avg,
            "test_loss_avg": test_loss_avg,
            "config_dump": asdict(cfg),
            # Legacy compatibility
            "args": cfg.extras.get("raw_cli", {}),
        }
        torch.save(ckpt, path)

    @classmethod
    def load(
        cls,
        *,
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: torch.device | str = "cpu",
        strict: bool = False,
    ) -> dict[str, Any]:
        raw = torch.load(path, map_location=map_location)
        if not isinstance(raw, dict):
            raise ValueError(f"Checkpoint at {path} must be dict, got {type(raw).__name__}")

        state = raw.get("model_state_dict", raw.get("state_dict", raw))
        if isinstance(state, dict):
            stripped = {}
            for k, v in state.items():
                key = k[7:] if isinstance(k, str) and k.startswith("module.") else k
                stripped[key] = v
            state = stripped
        missing, unexpected = model.load_state_dict(state, strict=strict)

        if optimizer is not None and isinstance(raw.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(raw["optimizer_state_dict"])

        return {
            "epoch": int(raw.get("epoch", 0)),
            "global_step": int(raw.get("global_step", 0)),
            "schema_version": int(raw.get("schema_version", 1)),
            "pipeline": raw.get("pipeline", "legacy"),
            "missing": missing,
            "unexpected": unexpected,
        }

