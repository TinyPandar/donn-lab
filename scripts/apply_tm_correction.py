from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from donn_lab.tm.checkpoint_phase import (
    apply_phase_correction_to_state,
    checkpoint_model_state,
    write_column_phase_csv,
)
from donn_lab.tm.drift import estimate_column_phase_delta
from donn_lab.tm.io import load_tmatrix_numpy, parse_shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate input-column phase drift from H0/H1 and write a corrected DONN checkpoint. "
            "The default correction assumes H1 ~= H0 @ diag(exp(i*delta)); phase masks are updated as phase-delta."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--h0", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--shape", type=str, default=None, help="Required for raw memmap input, e.g. 16384,16384.")
    parser.add_argument("--dtype", default="complex64")
    parser.add_argument("--layout", default="out_in", choices=["out_in", "in_out"])
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--phase-sign", default="subtract", choices=["subtract", "add"])
    parser.add_argument("--no-row-align", action="store_true", help="Skip row phase gauge removal before estimating columns.")
    parser.add_argument(
        "--gauge-iters",
        type=int,
        default=3,
        help="Alternating row/column phase-fit passes when row alignment is enabled.",
    )
    parser.add_argument("--no-wrap", action="store_true", help="Do not wrap corrected phase values into [0, 2pi).")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tm_phase_correction"))
    parser.add_argument("--out-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shape = parse_shape(args.shape)
    h0 = load_tmatrix_numpy(args.h0, shape=shape, dtype=args.dtype, layout=args.layout)
    h1 = load_tmatrix_numpy(args.h1, shape=shape, dtype=args.dtype, layout=args.layout)
    delta, corr, summary = estimate_column_phase_delta(
        h0,
        h1,
        chunk_rows=args.chunk_rows,
        row_align=not args.no_row_align,
        gauge_iters=args.gauge_iters,
    )

    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(raw_checkpoint, dict):
        raise ValueError(f"Checkpoint at {args.checkpoint} must be a dict, got {type(raw_checkpoint).__name__}")

    state = checkpoint_model_state(raw_checkpoint)
    applied = apply_phase_correction_to_state(
        state,
        delta,
        phase_sign=args.phase_sign,
        wrap=not args.no_wrap,
    )

    out_checkpoint = args.out_checkpoint or args.out_dir / f"{args.checkpoint.stem}_tm_corrected.pth"
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "corrected_checkpoint": str(out_checkpoint),
        "h0": str(args.h0),
        "h1": str(args.h1),
        "shape": summary["shape"],
        "layout": args.layout,
        "dtype": args.dtype,
        "phase_sign": args.phase_sign,
        "wrap": not args.no_wrap,
        "applied": applied,
        "drift_summary": summary,
    }
    raw_checkpoint["tm_phase_correction"] = metadata
    torch.save(raw_checkpoint, out_checkpoint)

    np.save(args.out_dir / "column_phase_delta_rad.npy", delta)
    write_column_phase_csv(delta, corr, args.out_dir / "column_phase_correction.csv")
    (args.out_dir / "tm_phase_correction_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
