from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface as si


# -----------------------------
# ACG utilities (paper version)
# -----------------------------

def acg_positive_lags_counts(
    spike_times_s: np.ndarray,
    bin_ms: float = 1.0,
    max_lag_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an autocorrelogram (ACG) for positive lags only: (0, max_lag_ms].
    Returns:
      centers_ms: bin centers in ms
      counts: counts of spike pairs per bin

    Implementation: for each spike, look forward until dt exceeds max_lag.
    """
    st = np.asarray(spike_times_s, dtype=float)
    st = st[np.isfinite(st)]
    st.sort()

    bins = np.arange(0.0, max_lag_ms + bin_ms, bin_ms)  # edges: 0..max
    centers = (bins[:-1] + bins[1:]) / 2

    if st.size < 2:
        return centers, np.zeros_like(centers, dtype=float)

    max_lag_s = max_lag_ms / 1000.0
    diffs_ms: List[float] = []

    # Forward differences bounded by max lag (O(N*k) but k small for 50 ms)
    for i in range(st.size):
        j = i + 1
        while j < st.size:
            dt = st[j] - st[i]
            if dt > max_lag_s:
                break
            diffs_ms.append(dt * 1000.0)
            j += 1

    diffs_ms = np.asarray(diffs_ms, dtype=float)
    counts, edges = np.histogram(diffs_ms, bins=bins)
    return centers, counts.astype(float)


def burst_index_paper(
    spike_times_s: np.ndarray,
    bin_ms: float = 1.0,
    max_lag_ms: float = 50.0,
    burst_window_ms: Tuple[float, float] = (0.0, 10.0),
    baseline_window_ms: Tuple[float, float] = (40.0, 50.0),
) -> Tuple[float, float, float]:
    """
    Paper definition:
      burst_index = (# pairs in 0–10 ms) / (# pairs in 40–50 ms)

    Returns:
      (burst_index, numerator, denominator)
    """
    centers, counts = acg_positive_lags_counts(spike_times_s, bin_ms=bin_ms, max_lag_ms=max_lag_ms)

    def wsum(w0: float, w1: float) -> float:
        mask = (centers >= w0) & (centers < w1)
        return float(np.sum(counts[mask]))

    num = wsum(*burst_window_ms)
    den = wsum(*baseline_window_ms)

    if den <= 0:
        return np.nan, num, den

    return num / den, num, den


# -----------------------------
# Legacy metric (matches pasted code style)
# -----------------------------

def interspike_histogram_legacy(spkTr1_s: np.ndarray, spkTr2_s: np.ndarray, maxInt_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy (from pasted code):
    - converts to ms
    - uses n_divisions=50
    - computes all pairwise differences via shifting (O(N^2), expensive for big N)
    - returns symmetric histogram in [-maxInt, +maxInt]
    """
    spkTr1 = np.asarray(spkTr1_s, dtype=float) * 1000.0
    spkTr2 = np.asarray(spkTr2_s, dtype=float) * 1000.0
    n_divisions = 50

    nSpk = len(spkTr1)
    if nSpk == 0 or len(spkTr2) == 0:
        binwidth = maxInt_ms / n_divisions
        bins = np.arange(-maxInt_ms, maxInt_ms + binwidth, binwidth)
        counts = np.full(len(bins) - 1, np.nan)
        centers = bins[:-1] + binwidth / 2
        return centers, counts

    int_matrix = np.zeros((nSpk, nSpk - 1), dtype=float)
    for ii in range(1, nSpk):
        shifted = np.roll(spkTr2, ii)
        int_matrix[:, ii - 1] = spkTr1 - shifted

    binwidth = maxInt_ms / n_divisions
    bins = np.arange(-maxInt_ms, maxInt_ms + binwidth, binwidth)
    counts, _ = np.histogram(int_matrix.flatten(), bins=bins)
    centers = bins[:-1] + binwidth / 2
    return centers, counts.astype(float)


def burst_index_legacy_like_code(
    spike_times_s: np.ndarray,
    sample_down_n: int = 10000,
) -> float:
    """
    Implements the same logic as the pasted snippet:
      max_peak_10 = max(cou[51:61])
      mean_val_40_50 = mean(cou[90:])
      burst_index_temp = max_peak_10 - mean_val_40_50
      normalized by max_peak_10 or mean_val_40_50

    NOTE: This is *not* the paper definition. It's just for comparison.
    """
    st = np.asarray(spike_times_s, dtype=float)
    st = st[np.isfinite(st)]
    st.sort()
    if st.size < 2:
        return np.nan

    # match their speed hack
    if st.size > sample_down_n:
        st = st[:sample_down_n]

    bins, cou = interspike_histogram_legacy(st, st, maxInt_ms=50.0)

    # mimic their indexing assumptions (0 lag around middle)
    # For maxInt=50, n_divisions=50 => binwidth=1ms; centers length ~100
    if len(cou) < 95:
        return np.nan

    max_peak_10 = np.max(cou[51:61])
    mean_val_40_50 = np.mean(cou[90:])

    burst_index_temp = max_peak_10 - mean_val_40_50

    if max_peak_10 == 0 or mean_val_40_50 == 0:
        return np.nan

    if burst_index_temp > 0:
        return float(burst_index_temp / max_peak_10)
    elif burst_index_temp < 0:
        return float(burst_index_temp / mean_val_40_50)
    else:
        return 0.0


# -----------------------------
# Main audit runner
# -----------------------------

def _get_sampling_frequency_from_analyzer(sorting_analyzer) -> float:
    # be defensive across SI versions
    try:
        return float(sorting_analyzer.sampling_frequency)
    except Exception:
        pass
    try:
        return float(sorting_analyzer.recording.get_sampling_frequency())
    except Exception:
        pass
    return 30000.0


def plot_acg_diagnostic(
    unit_id: int,
    spike_times_s: np.ndarray,
    out_path: Path,
    bin_ms: float = 1.0,
    max_lag_ms: float = 50.0,
) -> None:
    centers, counts = acg_positive_lags_counts(spike_times_s, bin_ms=bin_ms, max_lag_ms=max_lag_ms)
    bw = bin_ms

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(centers, counts, width=bw, align="center")
    ax.axvspan(0, 10, alpha=0.2, label="0–10 ms")
    ax.axvspan(40, 50, alpha=0.2, label="40–50 ms")
    ax.set_title(f"Unit {unit_id} ACG (positive lags)")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, max_lag_ms)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def audit_burst_index(
    derivatives_base: str,
    project_base: str,
    bin_ms: float = 1.0,
    max_lag_ms: float = 50.0,
    make_plots: bool = True,
    n_plot_units_each: int = 5,
) -> Path:
    """
    READS from derivatives_base (no modifications).
    WRITES outputs to project_base (Chiara folder).

    Outputs:
      - burst_index_audit.csv
      - diagnostic plots for top/bottom units by paper burst index (optional)
    """
    # ---- INPUT (READ ONLY) ----
    unit_features_path = os.path.join(
        derivatives_base, "analysis", "cell_characteristics", "unit_features"
    )
    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")

    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Missing sorting_analyzer:\n{analyzer_path}")

    print("Loading sorting_analyzer (read-only):")
    print(" ", analyzer_path)
    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_path)

    sorting = sorting_analyzer.sorting
    fs_hz = _get_sampling_frequency_from_analyzer(sorting_analyzer)
    print("Sampling frequency (Hz):", fs_hz)

    # ---- OUTPUT (WRITE HERE) ----
    out_dir = Path(project_base) / "audits" / "burst_index"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "burst_index_audit.csv"

    # -----------------------------
    # Compute burst indices
    # -----------------------------
    rows = []
    for uid in sorting.unit_ids:
        st_samples = sorting.get_unit_spike_train(unit_id=uid)
        st_s = np.asarray(st_samples, dtype=float) / fs_hz

        bi_paper, num, den = burst_index_paper(
            st_s,
            bin_ms=bin_ms,
            max_lag_ms=max_lag_ms,
            burst_window_ms=(0.0, 10.0),
            baseline_window_ms=(40.0, 50.0),
        )

        bi_legacy = burst_index_legacy_like_code(st_s)

        rows.append(
            dict(
                unit_id=int(uid),
                n_spikes=int(len(st_s)),
                burst_index_paper=bi_paper,
                acg_pairs_0_10=num,
                acg_pairs_40_50=den,
                burst_index_legacy_normdiff=bi_legacy,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\nSaved audit CSV to:")
    print(" ", out_csv)

    # -----------------------------
    # Summary
    # -----------------------------
    finite = df[np.isfinite(df["burst_index_paper"])]
    print("\nPaper burst index summary (finite only):")
    if len(finite) > 0:
        print(finite["burst_index_paper"].describe())

    # -----------------------------
    # Diagnostic plots
    # -----------------------------
    if make_plots and len(finite) > 0:
        top = (
            finite.sort_values("burst_index_paper", ascending=False)
            .head(n_plot_units_each)["unit_id"]
            .tolist()
        )
        bot = (
            finite.sort_values("burst_index_paper", ascending=True)
            .head(n_plot_units_each)["unit_id"]
            .tolist()
        )
        plot_units = top + bot

        for uid in plot_units:
            st_samples = sorting.get_unit_spike_train(unit_id=int(uid))
            st_s = np.asarray(st_samples, dtype=float) / fs_hz

            plot_acg_diagnostic(
                unit_id=int(uid),
                spike_times_s=st_s,
                out_path=fig_dir / f"unit_{int(uid):04d}_acg_0_50ms.png",
                bin_ms=bin_ms,
                max_lag_ms=max_lag_ms,
            )

        print(f"\nSaved {len(plot_units)} diagnostic plots to:")
        print(" ", fig_dir)

    return out_csv



if __name__ == "__main__":

    # ---- READ ONLY INPUT ----
    derivatives_base = (
        r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task"
        r"\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    )

    # ---- WRITE OUTPUTS HERE ----
    project_base = (
        r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\cell_classification_projects\HCT"
        r"\sub-002_id-1R\ses-02_date-11092025"
    )

    audit_burst_index(
        derivatives_base=derivatives_base,
        project_base=project_base,
        bin_ms=1.0,
        max_lag_ms=50.0,
        make_plots=True,
        n_plot_units_each=5,
    )
