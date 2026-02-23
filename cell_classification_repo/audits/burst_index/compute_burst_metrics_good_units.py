from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import spikeinterface as si


def _get_fs_hz_from_analyzer(sorting_analyzer) -> float:
    """Robustly get sampling frequency (Hz) across SpikeInterface versions."""
    try:
        return float(sorting_analyzer.sampling_frequency)
    except Exception:
        pass
    try:
        return float(sorting_analyzer.recording.get_sampling_frequency())
    except Exception:
        pass
    return 30000.0


def _load_good_units_from_cluster_group(derivatives_base: str) -> np.ndarray:
    """Read Phy curation cluster_group.tsv and return good cluster IDs."""
    tsv_path = os.path.join(
        derivatives_base,
        "ephys",
        "concat_run",
        "sorting",
        "sorter_output",
        "cluster_group.tsv",
    )
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Missing cluster_group.tsv:\n{tsv_path}")

    df = pd.read_csv(tsv_path, sep="\t")
    if "group" not in df.columns or "cluster_id" not in df.columns:
        raise ValueError(f"Unexpected columns in cluster_group.tsv: {list(df.columns)}")

    good = df.loc[df["group"] == "good", "cluster_id"].to_numpy(dtype=int)
    return good


def frac_spikes_with_next_isi_below(
    spike_times_s: np.ndarray, threshold_ms: float = 2.0
) -> Tuple[float, int, int]:
    """Fraction of spikes whose NEXT ISI is below threshold_ms."""
    st = np.asarray(spike_times_s, dtype=float)
    st = st[np.isfinite(st)]
    st.sort()

    n = int(st.size)
    if n < 2:
        return np.nan, 0, n

    isi_ms = np.diff(st) * 1000.0
    v = int(np.sum(isi_ms < threshold_ms))
    return float(v / n), v, n


def autocorrelogram_histogram(
    spike_times_s: np.ndarray,
    bin_ms: float = 1.0,
    max_lag_ms: float = 50.0,
    positive_lags_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ACG histogram from spike-time differences.
    If positive_lags_only=True: uses only positive lags in [0, max_lag_ms)
    Else: symmetric histogram in [-max_lag_ms, +max_lag_ms)
    Returns: (centers_ms, counts)
    """
    st_ms = np.asarray(spike_times_s, dtype=float) * 1000.0
    st_ms = st_ms[np.isfinite(st_ms)]
    st_ms.sort()

    if positive_lags_only:
        edges = np.arange(0.0, max_lag_ms + bin_ms, bin_ms)
    else:
        edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)

    centers = edges[:-1] + bin_ms / 2.0

    if st_ms.size < 2:
        # Use zeros (not NaNs) to avoid RuntimeWarnings downstream
        return centers, np.zeros_like(centers, dtype=float)

    # accumulate positive differences within max lag (efficient for small max_lag)
    diffs = []
    n = st_ms.size
    for i in range(n):
        j = i + 1
        while j < n:
            d = st_ms[j] - st_ms[i]
            if d >= max_lag_ms:
                break
            diffs.append(d)
            j += 1
    diffs = np.asarray(diffs, dtype=float)

    if positive_lags_only:
        counts, _ = np.histogram(diffs, bins=edges)
        return centers, counts.astype(float)
    else:
        all_diffs = np.concatenate([-diffs, diffs])
        counts, _ = np.histogram(all_diffs, bins=edges)
        return centers, counts.astype(float)


def burst_contrast_from_acg(bins_ms: np.ndarray, counts: np.ndarray) -> float:
    """Legacy/shape-sensitive burst contrast metric."""
    bins = np.asarray(bins_ms, dtype=float)
    cou = np.asarray(counts, dtype=float)

    win_0_10 = (bins >= 0.0) & (bins < 10.0)
    win_40_50 = (bins >= 40.0) & (bins < 50.0)
    if (not np.any(win_0_10)) or (not np.any(win_40_50)):
        return np.nan

    max_peak_10 = float(np.nanmax(cou[win_0_10]))
    mean_val_40_50 = float(np.nanmean(cou[win_40_50]))
    if (not np.isfinite(max_peak_10)) or (not np.isfinite(mean_val_40_50)) or (max_peak_10 == 0):
        return np.nan

    temp = max_peak_10 - mean_val_40_50
    if temp > 0:
        return float(temp / max_peak_10)
    elif temp < 0:
        return float(temp / mean_val_40_50) if mean_val_40_50 != 0 else np.nan
    else:
        return 0.0


def burst_index_paper_from_acg(
    bins_ms: np.ndarray, counts: np.ndarray
) -> Tuple[float, float, float]:
    """Paper burst index: sum(0–10 ms) / sum(40–50 ms)."""
    bins = np.asarray(bins_ms, dtype=float)
    cou = np.asarray(counts, dtype=float)

    win_0_10 = (bins >= 0.0) & (bins < 10.0)
    win_40_50 = (bins >= 40.0) & (bins < 50.0)
    if (not np.any(win_0_10)) or (not np.any(win_40_50)):
        return np.nan, np.nan, np.nan

    num = float(np.nansum(cou[win_0_10]))
    den = float(np.nansum(cou[win_40_50]))
    if (not np.isfinite(num)) or (not np.isfinite(den)) or (den <= 0):
        return np.nan, num, den

    return float(num / den), num, den


def normalize_ratio_0_1(r: float) -> float:
    """Squash nonnegative ratio r into [0,1): r/(1+r)."""
    if (not np.isfinite(r)) or (r < 0):
        return np.nan
    return float(r / (1.0 + r))


def compute_burst_metrics_good_units(
    derivatives_base: str,
    project_base: str,
    bin_ms: float = 1.0,
    max_lag_ms: float = 50.0,
    positive_lags_only: bool = True,
    qc_threshold_ms: float = 2.0,
    qc_exclude_frac: float = 0.01,
) -> Path:
    # ---- INPUT (read-only) ----
    unit_features_path = os.path.join(
        derivatives_base, "analysis", "cell_characteristics", "unit_features"
    )
    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Missing sorting_analyzer:\n{analyzer_path}")

    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_path)
    sorting = sorting_analyzer.sorting
    fs_hz = _get_fs_hz_from_analyzer(sorting_analyzer)

    good_ids = set(map(int, _load_good_units_from_cluster_group(derivatives_base)))
    unit_ids = [int(u) for u in sorting.unit_ids if int(u) in good_ids]
    print(f"GOOD units: {len(unit_ids)}")

    # ---- OUTPUT (write here) ----
    out_dir = Path(project_base) / "outputs" / "burst_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "burst_metrics_good_units.csv"

    rows = []
    for uid in unit_ids:
        st_samples = sorting.get_unit_spike_train(unit_id=int(uid))
        st_s = np.asarray(st_samples, dtype=float) / fs_hz

        frac2, nviol2, nspk = frac_spikes_with_next_isi_below(
            st_s, threshold_ms=qc_threshold_ms
        )
        paper_excl = bool(np.isfinite(frac2) and (frac2 > qc_exclude_frac))

        bins_ms, counts = autocorrelogram_histogram(
            st_s,
            bin_ms=bin_ms,
            max_lag_ms=max_lag_ms,
            positive_lags_only=positive_lags_only,
        )
        bi, num, den = burst_index_paper_from_acg(bins_ms, counts)
        bi01 = normalize_ratio_0_1(bi)
        bc = burst_contrast_from_acg(bins_ms, counts)

        rows.append(
            dict(
                unit_id=int(uid),
                n_spikes=int(nspk),
                n_nextISI_lt_2ms=int(nviol2),
                frac_nextISI_lt_2ms=float(frac2) if np.isfinite(frac2) else np.nan,
                paper_exclude_2ms=paper_excl,
                burst_index=bi,
                burst_index_norm01=bi01,
                acg_num_0_10=num,
                acg_den_40_50=den,
                burst_contrast=bc,
            )
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return out_csv


if __name__ == "__main__":
    derivatives_base = (
        r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task"
        r"\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    )
    project_base = (
        r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\cell_classification_projects\HCT"
        r"\sub-002_id-1R\ses-02_date-11092025"
    )

    compute_burst_metrics_good_units(
        derivatives_base=derivatives_base,
        project_base=project_base,
        bin_ms=1.0,
        max_lag_ms=50.0,
        positive_lags_only=True,
    )
