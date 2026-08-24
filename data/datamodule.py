from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from . import data_loader

from config.schema import ExperimentConfig
from registry import register_dataset


class DataModule(Protocol):
    def setup(self, cfg: ExperimentConfig) -> None:
        ...

    def train_iter(self) -> Iterable:
        ...

    def val_iter(self) -> Iterable:
        ...

    def estimate_batches(self, split: str) -> int | None:
        ...

    def sample_for_vis(self, split: str, n: int) -> list:
        ...


@dataclass
class LegacyDataModule:
    cfg: ExperimentConfig

    def setup(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        data_loader.update_config(
            h_in=int(cfg.data.h_in),
            w_in=int(cfg.data.w_in),
            h_out=int(cfg.data.h_out),
            w_out=int(cfg.data.w_out),
            aug_params={
                "enable_aug": bool(cfg.data.enable_aug),
                "input_mode": str(cfg.data.input_mode),
                "input_mode_explicit": bool(cfg.extras.get("data_input_mode_explicit", False)),
                "aug_hflip_p": float(cfg.data.aug_hflip_p),
                "aug_color_p": float(cfg.data.aug_color_p),
                "aug_blur_p": float(cfg.data.aug_blur_p),
                "aug_noise_p": float(cfg.data.aug_noise_p),
                "zoom_crop_mode": str(cfg.data.zoom_crop_mode),
                "zoom_crop_factor": float(cfg.data.zoom_crop_factor),
                "vehicle_channel_mode": str(cfg.data.vehicle_channel_mode),
                "vehicle_channel": str(cfg.data.vehicle_channel),
                "vehicle_channel_invert": bool(cfg.data.vehicle_channel_invert),
                "vehicle_channel_p_low": float(cfg.data.vehicle_channel_p_low),
                "vehicle_channel_p_high": float(cfg.data.vehicle_channel_p_high),
                "vehicle_target_mode": str(cfg.data.vehicle_target_mode),
                "vehicle_target_column": str(cfg.data.vehicle_target_column),
                "mnist_target_mode": str(cfg.data.mnist_target_mode),
            },
        )

    def _iter(self, split: str):
        return data_loader.get_batch_iter(
            dataset_name=self.cfg.dataset,
            data_root=self.cfg.data_root,
            split=split,
            batch_size=int(self.cfg.data.batch_size),
            label_filter=str(self.cfg.data.label_filter),
            multiple_objects=bool(self.cfg.data.multiple_objects),
        )

    def train_iter(self):
        return self._iter("Train")

    def val_iter(self):
        return self._iter("Test")

    def estimate_batches(self, split: str) -> int | None:
        max_batches = self.cfg.data.max_train_batches if split.lower().startswith("train") else self.cfg.data.max_test_batches
        return data_loader.estimate_total_batches(
            dataset_name=self.cfg.dataset,
            data_root=self.cfg.data_root,
            split=split,
            batch_size=int(self.cfg.data.batch_size),
            max_batches=int(max_batches),
            label_filter=str(self.cfg.data.label_filter),
            multiple_objects=bool(self.cfg.data.multiple_objects),
        )

    def sample_for_vis(self, split: str, n: int) -> list:
        samples: list = []
        want = int(n)
        if want <= 0:
            return samples
        for x_batch, target_batch in self._iter(split):
            for idx in range(int(x_batch.shape[0])):
                samples.append(
                    {
                        "x": x_batch[idx : idx + 1].detach().cpu(),
                        "target": target_batch[idx : idx + 1].detach().cpu(),
                    }
                )
                if len(samples) >= want:
                    return samples
        return samples


def _legacy_factory(cfg: ExperimentConfig) -> LegacyDataModule:
    dm = LegacyDataModule(cfg=cfg)
    dm.setup(cfg)
    return dm


def register_builtin_datasets() -> None:
    # Keep per-name registration for discoverability and nicer error messages.
    for name in ("inria", "pennfudan", "mnist", "syn", "synped", "vehicle", "tracking_vehicle", "syntrack", "tracking"):
        try:
            register_dataset(name, _legacy_factory)
        except ValueError:
            pass
