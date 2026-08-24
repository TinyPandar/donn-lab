from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        def add_scalar(self, *args, **kwargs):
            _ = args, kwargs

        def add_text(self, *args, **kwargs):
            _ = args, kwargs

        def add_figure(self, *args, **kwargs):
            _ = args, kwargs

        def flush(self):
            return None

        def close(self):
            return None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore[no-redef]
        _ = args, kwargs
        return iterable


def _progress_bar(iterable, *, desc: str, total: int):
    """Build a terminal-safe progress bar.

    The linked train/test workflow captures the trainer's output through a
    pipe.  In that situation tqdm's carriage-return refreshes are translated
    into separate log lines by Python's Windows text stream, producing two
    noisy lines per batch (and mojibake for the Unicode bar).  Keep dynamic
    progress for a real terminal only; epoch summaries remain visible in
    redirected logs.
    """
    interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        disable=not interactive,
        mininterval=1.0,
        dynamic_ncols=interactive,
        ascii=(os.name == "nt"),
    )

from config.schema import ExperimentConfig
from data import register_builtin_datasets
from engine.checkpoint import CheckpointIO
from engine.distributed import DistState, init_dist, is_rank0, maybe_barrier, teardown_dist
from engine.visualization import visualize_epoch_samples
from models import register_builtin_models
from models.factory import DistContext
from pipelines import register_builtin_pipelines
from pipelines.base import TrainingPipeline
from registry import create_dataset


class TrainerEngine:
    def __init__(self, cfg: ExperimentConfig, pipeline: TrainingPipeline, clearml_task: Any | None = None):
        self.cfg = cfg
        self.pipeline = pipeline
        self.clearml_task = clearml_task
        self.clearml_logger = clearml_task.get_logger() if clearml_task is not None else None
        self.dist_state: DistState | None = None
        self.writer: SummaryWriter | None = None
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.datamodule = None
        self.global_step = 0
        self.start_epoch = 1

    def _build_autocast(self) -> tuple[bool, torch.dtype]:
        state = self.dist_state
        assert state is not None
        if state.device.type != "cuda":
            return False, torch.float16
        if self.cfg.runtime.amp:
            return True, (torch.float16 if str(self.cfg.runtime.amp_dtype) == "fp16" else torch.bfloat16)
        t_dtype = str(self.cfg.model_cfg.tmatrix_compute_dtype)
        if t_dtype == "fp16":
            return True, torch.float16
        if t_dtype == "bf16":
            return True, torch.bfloat16
        return False, torch.float16

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if isinstance(model, DDP) else model

    def _report_clearml_scalar(self, title: str, series: str, value: float, iteration: int, *, is_batch: bool = False) -> None:
        if self.clearml_logger is None or not self.cfg.clearml.report_scalars:
            return
        if is_batch and not self.cfg.clearml.report_batch_scalars:
            return
        self.clearml_logger.report_scalar(title=title, series=series, value=value, iteration=iteration)

    def _broadcast_batch_if_needed(self, x_batch: torch.Tensor, target_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.dist_state
        assert state is not None
        if state.is_dist:
            dist.broadcast(x_batch, src=0)
            dist.broadcast(target_batch, src=0)
        return x_batch, target_batch

    def _maybe_resume(self) -> None:
        assert self.model is not None and self.optimizer is not None and self.dist_state is not None
        ckpt_path = CheckpointIO.resolve_resume_path(self.cfg)
        if not ckpt_path:
            return
        if not os.path.isfile(ckpt_path):
            print(f"Warning: resume requested but checkpoint not found: {ckpt_path}")
            return
        info = CheckpointIO.load(
            path=ckpt_path,
            model=self._unwrap_model(self.model),
            optimizer=self.optimizer,
            map_location=self.dist_state.device,
            strict=False,
        )
        self.start_epoch = int(info["epoch"]) + 1
        self.global_step = int(info["global_step"])
        if is_rank0(self.dist_state):
            print(
                f"Loaded checkpoint {ckpt_path} (schema={info['schema_version']}, "
                f"pipeline={info['pipeline']}, missing={len(info['missing'])}, unexpected={len(info['unexpected'])})"
            )

    def _init_components(self) -> None:
        register_builtin_models()
        register_builtin_datasets()
        register_builtin_pipelines()

        self.dist_state = init_dist(self.cfg.runtime.device)
        state = self.dist_state
        assert state is not None
        rank0 = is_rank0(state)

        if self.cfg.runtime.memory_snapshot and state.device.type == "cuda" and rank0:
            torch.cuda.memory._record_memory_history(enabled="all", max_entries=100000)

        self.datamodule = create_dataset(self.cfg.dataset, self.cfg)

        dist_ctx = DistContext(is_dist=state.is_dist, local_rank=state.local_rank, world_size=state.world_size)
        model = self.pipeline.build_model(self.cfg, state.device, dist_ctx)
        if state.is_dist:
            model = DDP(
                model,
                device_ids=([state.local_rank] if state.device.type == "cuda" else None),
                output_device=(state.local_rank if state.device.type == "cuda" else None),
                find_unused_parameters=False,
                broadcast_buffers=False,
            )
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg.optim.lr))
        self.pipeline.setup(self.cfg, self._unwrap_model(self.model), state.device, dist_ctx)

        self._maybe_resume()

        if rank0:
            self.writer = SummaryWriter(self.cfg.logging.log_dir, comment=self.cfg.logging.comment)
            with open(os.path.join(self.cfg.logging.log_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(self.cfg), f, indent=2, ensure_ascii=False)
            self.writer.add_text("Config/Resolved", f"```json\n{json.dumps(asdict(self.cfg), indent=2, ensure_ascii=False)}\n```", 0)
            for dep in self.cfg.extras.get("deprecated_flags", []):
                print(f"Warning: deprecated flag used: {dep}; compatibility mode is enabled.")

    def run(self) -> None:
        self.pipeline.validate_config(self.cfg)
        self._init_components()

        assert self.dist_state is not None and self.model is not None and self.optimizer is not None and self.datamodule is not None
        state = self.dist_state
        rank0 = is_rank0(state)

        ac_enabled, ac_dtype = self._build_autocast()
        scaler = GradScaler("cuda", enabled=(ac_enabled and ac_dtype == torch.float16))

        for epoch in range(self.start_epoch, int(self.cfg.optim.epochs) + 1):
            self.model.train()
            t0 = time.time()
            epoch_loss_sum = 0.0
            epoch_batch_count = 0

            batch_iter = self.datamodule.train_iter()
            if rank0:
                total = self.datamodule.estimate_batches("Train")
                pbar = _progress_bar(
                    batch_iter,
                    desc=f"Epoch {epoch}/{self.cfg.optim.epochs}",
                    total=total,
                )
            else:
                pbar = batch_iter

            for b_idx, (x_batch, target_batch) in enumerate(pbar, start=1):
                x_batch = x_batch.to(state.device, non_blocking=True)
                target_batch = target_batch.to(state.device, non_blocking=True)
                x_batch, target_batch = self._broadcast_batch_if_needed(x_batch, target_batch)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast("cuda", enabled=ac_enabled, dtype=ac_dtype if ac_enabled else None):
                    out = self.pipeline.training_step((x_batch, target_batch), self.model, self.cfg)
                loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if self.cfg.runtime.memory_snapshot and rank0 and state.device.type == "cuda" and b_idx == int(self.cfg.runtime.memory_snapshot_batch):
                    snapshot_path = os.path.join(self.cfg.logging.log_dir, f"memory_snapshot_rank{state.rank}_batch{b_idx}.pickle")
                    torch.cuda.memory._dump_snapshot(snapshot_path)

                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value
                epoch_batch_count += 1

                if self.writer is not None:
                    self.writer.add_scalar("train/loss_batch", loss_value, self.global_step)
                    for k, v in out.metrics.items():
                        self.writer.add_scalar(f"train/{k}_batch", float(v), self.global_step)
                if rank0:
                    self._report_clearml_scalar("train", "loss_batch", loss_value, self.global_step, is_batch=True)
                    for k, v in out.metrics.items():
                        self._report_clearml_scalar("train", f"{k}_batch", float(v), self.global_step, is_batch=True)
                self.global_step += 1

                if rank0 and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        avg=f"{epoch_loss_sum / max(epoch_batch_count, 1):.4f}",
                        refresh=False,
                    )

                if self.cfg.data.max_train_batches > 0 and b_idx >= int(self.cfg.data.max_train_batches):
                    break

            if rank0 and hasattr(pbar, "close"):
                pbar.close()

            if epoch_batch_count == 0:
                if rank0:
                    print(f"[Epoch {epoch}] no training batches produced")
                continue

            epoch_loss_avg = epoch_loss_sum / epoch_batch_count
            epoch_time = time.time() - t0
            if rank0:
                print(f"[Epoch {epoch}/{self.cfg.optim.epochs}] train_avg_loss={epoch_loss_avg:.6f} batches={epoch_batch_count} time={epoch_time:.1f}s")
                if self.writer is not None:
                    self.writer.add_scalar("train/loss_epoch", epoch_loss_avg, epoch)
                self._report_clearml_scalar("train", "loss_epoch", epoch_loss_avg, epoch)

            maybe_barrier(state)
            upload_final_visualization = rank0 and epoch == int(self.cfg.optim.epochs)
            visualize_epoch_samples(
                model=self._unwrap_model(self.model),
                datamodule=self.datamodule,
                cfg=self.cfg,
                epoch=epoch,
                writer=self.writer if rank0 else None,
                clearml_logger=self.clearml_logger if upload_final_visualization else None,
            )
            maybe_barrier(state)

            self.model.eval()
            test_loss_sum = 0.0
            test_batch_count = 0
            test_metric_sums: dict[str, float] = {}
            test_metric_counts: dict[str, int] = {}
            with torch.no_grad():
                test_iter = self.datamodule.val_iter()
                if rank0:
                    total_test = self.datamodule.estimate_batches("Test")
                    test_pbar = _progress_bar(
                        test_iter,
                        desc=f"Testing Epoch {epoch}",
                        total=total_test,
                    )
                else:
                    test_pbar = test_iter

                for tb_idx, (x_batch, target_batch) in enumerate(test_pbar, start=1):
                    x_batch = x_batch.to(state.device, non_blocking=True)
                    target_batch = target_batch.to(state.device, non_blocking=True)
                    x_batch, target_batch = self._broadcast_batch_if_needed(x_batch, target_batch)

                    with autocast("cuda", enabled=ac_enabled, dtype=ac_dtype if ac_enabled else None):
                        vout = self.pipeline.validation_step((x_batch, target_batch), self.model, self.cfg)
                    vloss = float(vout.loss.detach().cpu())
                    test_loss_sum += vloss
                    test_batch_count += 1
                    for k, v in vout.metrics.items():
                        if k == "val_loss":
                            continue
                        test_metric_sums[k] = test_metric_sums.get(k, 0.0) + float(v)
                        test_metric_counts[k] = test_metric_counts.get(k, 0) + 1

                    if rank0 and hasattr(test_pbar, "set_postfix"):
                        test_pbar.set_postfix(
                            loss=f"{vloss:.4f}",
                            avg=f"{test_loss_sum / max(test_batch_count, 1):.4f}",
                            refresh=False,
                        )
                    if self.cfg.data.max_test_batches > 0 and tb_idx >= int(self.cfg.data.max_test_batches):
                        break

                if rank0 and hasattr(test_pbar, "close"):
                    test_pbar.close()

            maybe_barrier(state)
            test_loss_avg = (test_loss_sum / test_batch_count) if test_batch_count > 0 else None
            test_metric_avgs = {
                k: (test_metric_sums[k] / max(test_metric_counts.get(k, 0), 1))
                for k in sorted(test_metric_sums)
            }
            if rank0:
                if test_loss_avg is None:
                    print("[Eval] no validation batches produced")
                else:
                    print(f"[Epoch {epoch}/{self.cfg.optim.epochs}] test_avg_loss={test_loss_avg:.6f} batches={test_batch_count}")
                    if self.writer is not None:
                        self.writer.add_scalar("test/loss_epoch", test_loss_avg, epoch)
                    self._report_clearml_scalar("test", "loss_epoch", test_loss_avg, epoch)
                    for k, v in test_metric_avgs.items():
                        print(f"[Epoch {epoch}/{self.cfg.optim.epochs}] test_{k}={v:.6f}")
                        if self.writer is not None:
                            self.writer.add_scalar(f"test/{k}", v, epoch)
                        self._report_clearml_scalar("test", k, v, epoch)

            save_now = (epoch % int(self.cfg.optim.save_every) == 0) or (epoch == int(self.cfg.optim.epochs))
            if rank0 and save_now:
                ckpt_path = os.path.join(self.cfg.logging.ckpt_dir, f"epoch_{epoch:03d}.pth")
                CheckpointIO.save(
                    path=ckpt_path,
                    epoch=epoch,
                    global_step=self.global_step,
                    model_state=self._unwrap_model(self.model).state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    train_loss_avg=epoch_loss_avg,
                    test_loss_avg=test_loss_avg,
                    cfg=self.cfg,
                )
                print(f"Saved checkpoint to: {ckpt_path}")

            self.model.train()

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        teardown_dist(state)
