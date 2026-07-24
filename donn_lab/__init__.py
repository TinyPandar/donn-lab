"""Lightweight lab-side interfaces for measured-TM DONN experiments.

The package is intentionally small and file-oriented. Hardware code writes
measured matrices such as H0/H1, while training and evaluation code consumes
those files through stable interfaces.
"""

__all__ = ["tm", "hardware", "workflows"]

