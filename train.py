from __future__ import annotations

import os
import sys

from config import load_experiment_config


def _inject_pipeline_arg(argv: list[str], forced_pipeline: str | None) -> list[str]:
    if not forced_pipeline:
        return argv
    has_pipeline = any(a == "--pipeline" or a.startswith("--pipeline=") for a in argv)
    if has_pipeline:
        return argv
    return ["--pipeline", forced_pipeline, *argv]


def main(argv: list[str] | None = None, forced_pipeline: str | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    args = _inject_pipeline_arg(args, forced_pipeline)
    cfg, _ = load_experiment_config(args)

    from engine import TrainerEngine
    from engine.clearml_integration import init_clearml_task
    from pipelines import create_registered_pipeline, register_builtin_pipelines

    clearml_task = None
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"TensorBoard logdir: {cfg.logging.log_dir}")
        clearml_task = init_clearml_task(cfg)

    register_builtin_pipelines()
    pipeline = create_registered_pipeline(cfg.pipeline)
    trainer = TrainerEngine(cfg=cfg, pipeline=pipeline, clearml_task=clearml_task)
    try:
        trainer.run()
    finally:
        if clearml_task is not None:
            clearml_task.close()


if __name__ == "__main__":
    main()
