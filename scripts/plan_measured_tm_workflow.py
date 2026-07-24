from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from donn_lab.workflows import ClosedLoopPaths, build_closed_loop_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the measured-TM stage-0 closed-loop command sequence.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--h0", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--python-bin", default="/home/limingfei/miniforge3/envs/speckle/bin/python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ClosedLoopPaths(
        config=args.config,
        h0=args.h0,
        h1=args.h1,
        run_dir=args.run_dir,
        report_dir=args.report_dir,
        checkpoint=args.checkpoint,
    )
    for idx, step in enumerate(build_closed_loop_steps(paths, python_bin=args.python_bin, num_layers=args.num_layers), start=1):
        print(f"# {idx}. {step.name}")
        print(f"# {step.note}")
        print(shlex.join(step.command))
        print()


if __name__ == "__main__":
    main()
