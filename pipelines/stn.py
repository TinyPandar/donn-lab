from __future__ import annotations

from config.schema import ExperimentConfig

from .base import TrainingPipeline


class STNPipeline(TrainingPipeline):
    name = "stn"

    def validate_config(self, cfg: ExperimentConfig) -> None:
        if cfg.model not in ("scatter_tile", "stn"):
            raise ValueError("pipeline=stn requires model 'scatter_tile' (or alias 'stn')")

