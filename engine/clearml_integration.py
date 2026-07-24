from __future__ import annotations

import os
from typing import Any

from config.schema import ExperimentConfig, dataclass_to_dict


def _task_name(cfg: ExperimentConfig) -> str:
    if cfg.clearml.task_name:
        return cfg.clearml.task_name
    if cfg.output.run_name:
        return cfg.output.run_name
    if cfg.logging.comment:
        return cfg.logging.comment
    return os.path.basename(os.path.normpath(cfg.logging.log_dir)) or "train"


def init_clearml_task(cfg: ExperimentConfig) -> Any | None:
    if not cfg.clearml.enabled:
        return None

    try:
        from clearml import Task
    except ImportError:
        print("Warning: ClearML is enabled but the clearml package is not installed; continuing without ClearML.")
        return None

    if cfg.clearml.offline:
        try:
            Task.set_offline(offline_mode=True)
        except Exception as exc:
            print(f"Warning: failed to enable ClearML offline mode: {exc}")

    frameworks: dict[str, bool] = {
        "tensorboard": bool(cfg.clearml.auto_connect_tensorboard),
        "pytorch": bool(cfg.clearml.auto_connect_pytorch),
    }
    kwargs: dict[str, Any] = {
        "project_name": cfg.clearml.project_name,
        "task_name": _task_name(cfg),
        "auto_connect_frameworks": frameworks,
    }
    if cfg.clearml.output_uri:
        kwargs["output_uri"] = cfg.clearml.output_uri

    try:
        try:
            task = Task.init(**kwargs)
        except TypeError:
            kwargs.pop("auto_connect_frameworks", None)
            task = Task.init(**kwargs)
        task.connect(dataclass_to_dict(cfg), name="config")
        if cfg.clearml.tags:
            task.set_tags(cfg.clearml.tags)
        task.get_logger().report_text(f"TensorBoard logdir: {cfg.logging.log_dir}")
        return task
    except Exception as exc:
        print(f"Warning: failed to initialize ClearML; continuing without ClearML. Error: {exc}")
        return None
