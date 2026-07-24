from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistState:
    is_dist: bool
    local_rank: int
    world_size: int
    rank: int
    device: torch.device


def init_dist(device_hint: str) -> DistState:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1
    if is_dist:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        device = torch.device(device_hint)
    return DistState(is_dist=is_dist, local_rank=local_rank, world_size=world_size, rank=rank, device=device)


def is_rank0(state: DistState) -> bool:
    return (not state.is_dist) or state.rank == 0


def maybe_barrier(state: DistState) -> None:
    if state.is_dist and dist.is_initialized():
        if state.device.type == "cuda":
            dist.barrier(device_ids=[state.local_rank])
        else:
            dist.barrier()


def teardown_dist(state: DistState) -> None:
    if state.is_dist and dist.is_initialized():
        maybe_barrier(state)
        dist.destroy_process_group()

