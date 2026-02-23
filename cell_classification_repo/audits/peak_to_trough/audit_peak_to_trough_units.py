from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import spikeinterface as si


def best_channel_mean_waveform(sorting_analyzer, unit_id: int) -> Tuple[np.ndarray, int]:
    """
    Returns (mean_waveform_1d, best_channel_index) for a unit.
    Best channel = largest peak-to-peak range on the unit's mean waveform.
    """
    waveforms_ext = sorting_analyzer.get_extension("waveforms")
    wf = waveforms_ext.get_waveforms_one_unit(unit_id=unit_id)

    if wf is None or len(wf) == 0:
        return np.array([]), -1

    mean_wf = wf.mean(axis=0)  # (n_samples, n_channels)
    ptp = mean_wf.max(axis=0) - mean_wf.min(axis=0)
    best_ch = int(np.argmax(ptp))
    return mean_wf[:, best_ch], best_ch


def valley_to_following_peak_ms(wave_1d: np.ndarray, fs_hz: float) -> Tuple[float, int, int]:
    """
    Valley (trough) -> following peak time in ms.
    Convention:
      - valley = minimum point
      - peak = maximum AFTER the valley
    Returns: (dt_ms, valley_index, peak_index)
    """
    if wave_1d.ndim != 1 or wave_1d.size < 5:
        return np.nan, -1, -1

    valley_i = int(np.argmin(wave_1d))
    post = wave_1d[valley_i:]
    if post.size < 2:
        return np.nan, valley_i, -1

    peak_rel = int(np.argmax(post))
    peak_i = valley_i + peak_rel

    dt_ms = (peak_i - valley_i) / fs_hz * 1000.0
    return float(dt_ms), valley_i, peak_i


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def audit_peak_to_trough(
    derivatives_base: str,
    project_base: str,
    fs_hz: float = 30000.0,
    peak_to_valley_col: str = "peak_to_valley",
) -> Path:
    """
    READS from derivatives_base (no modifications).
    WRITES outputs to project_base (Chiara folder).

    Compares:
      - waveform-derived valley->peak time (ms)
      - CSV peak_to_valley under 3 unit interpretations:
          1) seconds -> ms (x1000)
          2) already ms
          3) samples -> ms (/fs*1000)
    """

    # ---- INPUT (READ ONLY) ----
    unit_features_path = os.path.join(
        derivatives_base, "analysis", "cell_characteristics", "unit_features"
    )
    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")
    wf_csv = os.path.join(
        unit_features_path, "all_units_overview", "unit_waveform_metrics.csv"
    )

    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Missing sorting_analyzer:\n{analyzer_path}")
    if not os.path.exists(wf_csv):
        raise FileNotFoundError(f"Missing unit_waveform_metrics.csv:\n{wf_csv}")

    # ---- OUTPUT (WRITE HERE) ----
    out_dir = Path(project_base) / "audits" / "peak_to_trough"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "peak_to_trough_audit_report.csv"

    print("Loading sorting_analyzer (read-only):")
    print(" ", analyzer_path)
    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_path)

    if not sorting_analyzer.has_extension("waveforms"):
        raise RuntimeError(
            "sorting_analyzer does not have the 'waveforms' extension saved.\n"
            "This audit needs saved waveforms."
        )

    df = pd.read_csv(wf_csv)

    if "unit_ids" not in df.columns:
        raise ValueError("unit_ids column missing from unit_waveform_metrics.csv")
    if peak_to_valley_col not in df.columns:
        raise ValueError(
            f"{peak_to_valley_col} column missing from unit_waveform_metrics.csv.\n"
            f"Columns: {list(df.columns)}"
        )

    rows = []
    for _, r in df.iterrows():
        uid = _safe_int(r["unit_ids"])
        if uid is None:
            continue

        try:
            csv_val = float(r[peak_to_valley_col])
        except Exception:
            csv_val = np.nan

        mean_wf, best_ch = best_channel_mean_waveform(sorting_analyzer, uid)
        dt_ms, valley_i, peak_i = valley_to_following_peak_ms(mean_wf, fs_hz)

        rows.append(
            dict(
                unit_id=uid,
                best_channel=best_ch,
                waveform_valley_to_peak_ms=dt_ms,
                csv_peak_to_valley_raw=csv_val,
                csv_as_seconds_to_ms=csv_val * 1000.0,
                csv_as_ms=csv_val,
                csv_as_samples_to_ms=(csv_val / fs_hz) * 1000.0,
                valley_sample_index=valley_i,
                peak_sample_index=peak_i,
            )
        )

    audit = pd.DataFrame(rows)

    # Absolute errors vs waveform-derived value
    audit["err_if_seconds"] = (audit["csv_as_seconds_to_ms"] - audit["waveform_valley_to_peak_ms"]).abs()
    audit["err_if_ms"] = (audit["csv_as_ms"] - audit["waveform_valley_to_peak_ms"]).abs()
    audit["err_if_samples"] = (audit["csv_as_samples_to_ms"] - audit["waveform_valley_to_peak_ms"]).abs()

    # Which interpretation is best per unit?
    audit["best_match"] = audit[["err_if_seconds", "err_if_ms", "err_if_samples"]].idxmin(axis=1)

    audit.to_csv(out_csv, index=False)
    print("\nSaved audit report to:")
    print(" ", out_csv)

    print("\nBest-match counts:")
    print(audit["best_match"].value_counts(dropna=False))

    # Print a few examples (helps spot weirdness quickly)
    print("\nExample rows:")
    show = audit.sort_values("err_if_seconds").head(10)
    for _, rr in show.iterrows():
        print(
            f"unit {int(rr['unit_id'])} | waveform={rr['waveform_valley_to_peak_ms']:.4f} ms | "
            f"csv={rr['csv_peak_to_valley_raw']:.6g} | "
            f"csv*1000={rr['csv_as_seconds_to_ms']:.4f} | "
            f"best={rr['best_match']} | best_ch={int(rr['best_channel'])}"
        )

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

    audit_peak_to_trough(
        derivatives_base=derivatives_base,
        project_base=project_base,
        fs_hz=30000.0,
        peak_to_valley_col="peak_to_valley",
    )
