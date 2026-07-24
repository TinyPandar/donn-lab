from __future__ import annotations

from typing import Any

import numpy as np


def summarize_values(values: np.ndarray) -> dict[str, float]:
    """Return compact distribution statistics for report JSON files."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def compare_tmatrix_drift(
    h0: np.ndarray,
    h1: np.ndarray,
    *,
    chunk_rows: int = 256,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Compare two measured matrices without loading extra full-size copies.

    The comparison reports:
    - row gain/phase/correlation, useful for camera-side or reconstruction gauge drift
    - column gain/phase/correlation, useful for input-mode drift
    - raw and row-phase-aligned Frobenius differences
    """
    if h0.shape != h1.shape:
        raise ValueError(f"TM shape mismatch: H0={h0.shape}, H1={h1.shape}")
    n_out, n_in = h0.shape
    eps = 1e-12

    row_rows: list[dict[str, Any]] = []
    col_norm0_sq = np.zeros(n_in, dtype=np.float64)
    col_norm1_sq = np.zeros(n_in, dtype=np.float64)
    col_cross = np.zeros(n_in, dtype=np.complex128)

    fro0_sq = 0.0
    fro1_sq = 0.0
    raw_diff_sq = 0.0
    row_aligned_diff_sq = 0.0

    for start in range(0, n_out, int(chunk_rows)):
        end = min(start + int(chunk_rows), n_out)
        a = np.asarray(h0[start:end], dtype=np.complex64)
        b = np.asarray(h1[start:end], dtype=np.complex64)

        abs_a_sq = np.abs(a) ** 2
        abs_b_sq = np.abs(b) ** 2
        row_norm0 = np.sqrt(abs_a_sq.sum(axis=1, dtype=np.float64))
        row_norm1 = np.sqrt(abs_b_sq.sum(axis=1, dtype=np.float64))
        row_cross = np.sum(np.conj(a) * b, axis=1, dtype=np.complex128)
        row_corr = np.abs(row_cross) / np.maximum(row_norm0 * row_norm1, eps)
        row_phase = np.angle(row_cross)
        row_gain = row_norm1 / np.maximum(row_norm0, eps)

        row_phase_align = np.exp(-1j * row_phase).astype(np.complex64)[:, None]
        row_aligned_diff = b * row_phase_align - a

        fro0_sq += float(abs_a_sq.sum(dtype=np.float64))
        fro1_sq += float(abs_b_sq.sum(dtype=np.float64))
        raw_diff_sq += float((np.abs(b - a) ** 2).sum(dtype=np.float64))
        row_aligned_diff_sq += float((np.abs(row_aligned_diff) ** 2).sum(dtype=np.float64))

        col_norm0_sq += abs_a_sq.sum(axis=0, dtype=np.float64)
        col_norm1_sq += abs_b_sq.sum(axis=0, dtype=np.float64)
        col_cross += np.sum(np.conj(a) * b, axis=0, dtype=np.complex128)

        for offset in range(end - start):
            row_rows.append(
                {
                    "row": start + offset,
                    "norm_h0": float(row_norm0[offset]),
                    "norm_h1": float(row_norm1[offset]),
                    "gain_h1_over_h0": float(row_gain[offset]),
                    "phase_h1_vs_h0_rad": float(row_phase[offset]),
                    "phase_aligned_corr": float(row_corr[offset]),
                }
            )

    col_norm0 = np.sqrt(col_norm0_sq)
    col_norm1 = np.sqrt(col_norm1_sq)
    col_corr = np.abs(col_cross) / np.maximum(col_norm0 * col_norm1, eps)
    col_phase = np.angle(col_cross).astype(np.float32)
    col_gain = col_norm1 / np.maximum(col_norm0, eps)

    col_rows = [
        {
            "column": idx,
            "norm_h0": float(col_norm0[idx]),
            "norm_h1": float(col_norm1[idx]),
            "gain_h1_over_h0": float(col_gain[idx]),
            "phase_h1_vs_h0_rad": float(col_phase[idx]),
            "phase_aligned_corr": float(col_corr[idx]),
        }
        for idx in range(n_in)
    ]

    summary = {
        "shape": [int(n_out), int(n_in)],
        "relative_fro_norm_h1_over_h0": float(np.sqrt(fro1_sq) / max(np.sqrt(fro0_sq), eps)),
        "relative_raw_diff": float(np.sqrt(raw_diff_sq) / max(np.sqrt(fro0_sq), eps)),
        "relative_row_phase_aligned_diff": float(np.sqrt(row_aligned_diff_sq) / max(np.sqrt(fro0_sq), eps)),
        "row_corr": summarize_values(np.asarray([r["phase_aligned_corr"] for r in row_rows], dtype=np.float64)),
        "row_gain": summarize_values(np.asarray([r["gain_h1_over_h0"] for r in row_rows], dtype=np.float64)),
        "column_corr": summarize_values(col_corr),
        "column_gain": summarize_values(col_gain),
        "column_phase_delta_rad": summarize_values(col_phase),
    }
    return summary, row_rows, col_rows, col_phase


def estimate_column_phase_delta(
    h0: np.ndarray,
    h1: np.ndarray,
    *,
    chunk_rows: int = 256,
    row_align: bool = True,
    gauge_iters: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate input-column phase drift between H0 and H1.

    The model is H1 ~= R * H0 * D, where R is an output-row phase gauge and D is
    an input-column phase drift. Only D is returned for phase-mask correction.
    When `row_align` is true, the row and column phases are fitted by a few
    alternating passes to reduce GGS-style output gauge ambiguity.
    """
    if h0.shape != h1.shape:
        raise ValueError(f"TM shape mismatch: H0={h0.shape}, H1={h1.shape}")
    n_out, n_in = h0.shape
    eps = 1e-12

    col_phase_delta = np.zeros(n_in, dtype=np.float32)
    col_corr = np.zeros(n_in, dtype=np.float32)
    col_gain = np.ones(n_in, dtype=np.float64)
    row_corr_values: list[np.ndarray] = []

    n_iters = max(int(gauge_iters), 1) if row_align else 1
    for iter_idx in range(n_iters):
        col_norm0_sq = np.zeros(n_in, dtype=np.float64)
        col_norm1_sq = np.zeros(n_in, dtype=np.float64)
        col_cross = np.zeros(n_in, dtype=np.complex128)
        if iter_idx == n_iters - 1:
            row_corr_values = []

        col_phase_factor = np.exp(-1j * col_phase_delta).astype(np.complex64)[None, :]
        for start in range(0, n_out, int(chunk_rows)):
            end = min(start + int(chunk_rows), n_out)
            a = np.asarray(h0[start:end], dtype=np.complex64)
            b = np.asarray(h1[start:end], dtype=np.complex64)

            if row_align:
                row_norm0 = np.sqrt((np.abs(a) ** 2).sum(axis=1, dtype=np.float64))
                row_norm1 = np.sqrt((np.abs(b) ** 2).sum(axis=1, dtype=np.float64))
                row_cross = np.sum(np.conj(a) * b * col_phase_factor, axis=1, dtype=np.complex128)
                row_phase = np.angle(row_cross).astype(np.float32)
                if iter_idx == n_iters - 1:
                    row_corr_values.append(np.abs(row_cross) / np.maximum(row_norm0 * row_norm1, eps))
                b = b * np.exp(-1j * row_phase).astype(np.complex64)[:, None]

            col_norm0_sq += (np.abs(a) ** 2).sum(axis=0, dtype=np.float64)
            col_norm1_sq += (np.abs(b) ** 2).sum(axis=0, dtype=np.float64)
            col_cross += np.sum(np.conj(a) * b, axis=0, dtype=np.complex128)

        col_norm0 = np.sqrt(col_norm0_sq)
        col_norm1 = np.sqrt(col_norm1_sq)
        col_corr = (np.abs(col_cross) / np.maximum(col_norm0 * col_norm1, eps)).astype(np.float32)
        col_gain = col_norm1 / np.maximum(col_norm0, eps)
        col_phase_delta = np.angle(col_cross).astype(np.float32)

    summary: dict[str, Any] = {
        "shape": [int(n_out), int(n_in)],
        "row_align_before_column_estimate": bool(row_align),
        "gauge_iters": int(n_iters),
        "column_corr": summarize_values(col_corr),
        "column_gain_h1_over_h0": summarize_values(col_gain),
        "column_phase_delta_rad": summarize_values(col_phase_delta),
    }
    if row_corr_values:
        summary["row_corr_after_phase_only_fit"] = summarize_values(np.concatenate(row_corr_values))
    return col_phase_delta, col_corr, summary

