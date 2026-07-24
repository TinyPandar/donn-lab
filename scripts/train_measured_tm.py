from __future__ import annotations

import sys

from train import main


def _with_measured_tm_model(argv: list[str]) -> list[str]:
    has_model = any(arg == "--model" or arg.startswith("--model=") for arg in argv)
    if has_model:
        return argv
    return ["--model", "measured_tm_scatter", *argv]


if __name__ == "__main__":
    main(_with_measured_tm_model(sys.argv[1:]))
