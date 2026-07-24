from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from donn_lab.tm.drift import compare_tmatrix_drift
from donn_lab.tm.io import load_tmatrix_numpy, parse_shape, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two measured transmission matrices H0/H1.")
    parser.add_argument("--h0", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--shape", type=str, default=None, help="Required for raw memmap input, e.g. 16384,16384.")
    parser.add_argument("--dtype", default="complex64")
    parser.add_argument("--layout", default="out_in", choices=["out_in", "in_out"])
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tm_drift_compare"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shape = parse_shape(args.shape)
    h0 = load_tmatrix_numpy(args.h0, shape=shape, dtype=args.dtype, layout=args.layout)
    h1 = load_tmatrix_numpy(args.h1, shape=shape, dtype=args.dtype, layout=args.layout)
    summary, row_rows, col_rows, col_phase = compare_tmatrix_drift(h0, h1, chunk_rows=args.chunk_rows)

    (args.out_dir / "tm_drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv_rows(row_rows, args.out_dir / "tm_row_drift.csv")
    write_csv_rows(col_rows, args.out_dir / "tm_column_drift.csv")
    np.save(args.out_dir / "column_phase_delta_rad.npy", col_phase)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
