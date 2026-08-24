from __future__ import annotations

from registry import create_pipeline, register_pipeline

from .base import TrainingPipeline
from .classification import ClassificationPipeline
from .distill import DistillPipeline
from .stn import STNPipeline


def register_builtin_pipelines() -> None:
    entries = {
        "base": TrainingPipeline,
        "classification": ClassificationPipeline,
        "distill": DistillPipeline,
        "stn": STNPipeline,
    }
    for name, cls in entries.items():
        try:
            register_pipeline(name, cls)
        except ValueError:
            pass


def create_registered_pipeline(name: str):
    return create_pipeline(name)


__all__ = [
    "register_builtin_pipelines",
    "create_registered_pipeline",
    "TrainingPipeline",
    "ClassificationPipeline",
    "DistillPipeline",
    "STNPipeline",
]

