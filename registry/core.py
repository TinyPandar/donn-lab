from __future__ import annotations

from typing import Any, Callable

DATASET_REGISTRY: dict[str, Callable[..., Any]] = {}
MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}
PIPELINE_REGISTRY: dict[str, Callable[..., Any]] = {}


def _register(registry: dict[str, Callable[..., Any]], kind: str, name: str, factory: Callable[..., Any]) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError(f"{kind} name cannot be empty")
    if key in registry:
        raise ValueError(f"{kind} '{key}' already registered")
    registry[key] = factory


def register_dataset(name: str, factory: Callable[..., Any]) -> None:
    _register(DATASET_REGISTRY, "dataset", name, factory)


def register_model(name: str, factory: Callable[..., Any]) -> None:
    _register(MODEL_REGISTRY, "model", name, factory)


def register_pipeline(name: str, factory: Callable[..., Any]) -> None:
    _register(PIPELINE_REGISTRY, "pipeline", name, factory)


def _create(registry: dict[str, Callable[..., Any]], kind: str, name: str, *args, **kwargs) -> Any:
    key = str(name).strip().lower()
    if key not in registry:
        choices = ", ".join(sorted(registry.keys())) or "<empty>"
        raise KeyError(f"Unknown {kind} '{name}'. Available: {choices}")
    return registry[key](*args, **kwargs)


def create_dataset(name: str, *args, **kwargs) -> Any:
    return _create(DATASET_REGISTRY, "dataset", name, *args, **kwargs)


def create_model(name: str, *args, **kwargs) -> Any:
    return _create(MODEL_REGISTRY, "model", name, *args, **kwargs)


def create_pipeline(name: str, *args, **kwargs) -> Any:
    return _create(PIPELINE_REGISTRY, "pipeline", name, *args, **kwargs)

