from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import spikeinterface as si


# ----------------------------
# Helpers
# ----------------------------

def _get_fs_hz_from_analyzer(sorting_analyzer) -> float:
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
    """
    Reads Phy curation output cluster_group.tsv and returns good cluster IDs as ints.
    """
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
    if "cluster_id" not in df.columns or "KSLabel" not in df.columns:
        raise ValueError(f"Unexpected columns in cluster_group.tsv: {list(df.columns)}")

    good = df.loc[df["KSLabel"] == "good", "cluster_id"].to_numpy(dtype=int)
    return good


def frac_spikes_with_next_isi_below(
    spike_times_s: np.ndarray, threshold_ms: float = 2.0
) -> Tuple[float, int, int]:
    """
    Fraction of spikes whose NEXT inter-spike interval is below threshold.
    Returns (fraction, n_violations, n_spikes).
    """
    st = np.asarray(spike_times_s, dtype=float)
    st = st[np.isfinite(st)]
    st.sort()

    n = int(st.size)
    if n < 2:
        return np.nan, 0, n

    isi_ms = np.diff(st) * 1000.0
    v = int(np.sum(isi_ms < threshold_ms))
    frac = v / n
    return float(frac), v, n


# ----------------------------
# Pipeline step
# ----------------------------

def step_paper_isi_qc(
    derivatives_base: str,
    out_csv_all: Path,
    out_csv_good: Path,
    threshold_ms: float = 2.0,
    exclude_frac: float = 0.01,
) -> Tuple[Path, Path]:
    """
    Paper-style ISI QC audit (annotation only; not used for inclusion).

    Computes fraction of spikes whose NEXT ISI is < threshold_ms.
    Flags paper_exclude if fraction > exclude_frac.
    Adds meets_paper_isi = ~paper_exclude.

    Reads:
      derivatives_base/analysis/cell_characteristics/unit_features/spikeinterface_data
      derivatives_base/ephys/concat_run/sorting/sorter_output/cluster_group.tsv

    Writes:
      out_csv_good, out_csv_all
    """
    out_csv_all = Path(out_csv_all)
    out_csv_good = Path(out_csv_good)
    out_csv_all.parent.mkdir(parents=True, exist_ok=True)
    out_csv_good.parent.mkdir(parents=True, exist_ok=True)

    # ---- INPUT (read only) ----
    unit_features_path = os.path.join(
        derivatives_base, "analysis", "cell_characteristics", "unit_features"
    )
    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")

    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Missing sorting_analyzer:\n{analyzer_path}")

    good_units = _load_good_units_from_cluster_group(derivatives_base)

    print("Loading sorting_analyzer:")
    print(" ", analyzer_path)
    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_path)
    sorting = sorting_analyzer.sorting
    fs_hz = _get_fs_hz_from_analyzer(sorting_analyzer)
    print("Sampling frequency (Hz):", fs_hz)

    # Units present in sorting_analyzer
    all_units = np.array(list(map(int, sorting.unit_ids)), dtype=int)

    # Intersect GOOD with units actually present
    present = set(all_units.tolist())
    good_units = np.array([u for u in good_units if int(u) in present], dtype=int)

    def _compute_qc_df(unit_ids: np.ndarray) -> pd.DataFrame:
        rows = []
        for uid in unit_ids:
            st_samples = sorting.get_unit_spike_train(unit_id=int(uid))
            st_s = np.asarray(st_samples, dtype=float) / fs_hz

            frac, n_viol, n_spk = frac_spikes_with_next_isi_below(
                st_s, threshold_ms=threshold_ms
            )
            paper_exclude = bool(np.isfinite(frac) and frac > exclude_frac)
            rows.append(
                dict(
                    unit_id=int(uid),
                    n_spikes=int(n_spk),
                    n_nextISI_lt_threshold=int(n_viol),
                    frac_nextISI_lt_threshold=frac,
                    threshold_ms=float(threshold_ms),
                    exclude_frac=float(exclude_frac),
                    paper_exclude=paper_exclude,
                    meets_paper_isi=not paper_exclude,
                )
            )

        return pd.DataFrame(rows).sort_values(
            "frac_nextISI_lt_threshold", ascending=False
        )

    # ---- GOOD units ----
    df_good = _compute_qc_df(good_units)
    df_good.to_csv(out_csv_good, index=False)

    n_excl_good = int(df_good["paper_exclude"].sum())
    print("\nSaved paper ISI QC (GOOD units):")
    print(" ", out_csv_good)
    print(
        f"Good units: {len(df_good)} | Paper-excluded (>{exclude_frac*100:.1f}% under {threshold_ms} ms): {n_excl_good}"
    )

    # ---- ALL units ----
    df_all = _compute_qc_df(all_units)
    df_all.to_csv(out_csv_all, index=False)

    n_excl_all = int(df_all["paper_exclude"].sum())
    print("\nSaved paper ISI QC (ALL units):")
    print(" ", out_csv_all)
    print(
        f"All units: {len(df_all)} | Paper-excluded (>{exclude_frac*100:.1f}% under {threshold_ms} ms): {n_excl_all}"
    )

    return out_csv_all, out_csv_good


# ----------------------------
# CLI wrapper
# ----------------------------

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Paper-style ISI QC (2 ms rule) for ALL + GOOD units.")
    ap.add_argument("--derivatives_base", required=True)
    ap.add_argument("--out_csv_all", required=True, type=Path)
    ap.add_argument("--out_csv_good", required=True, type=Path)
    ap.add_argument("--threshold_ms", default=2.0, type=float)
    ap.add_argument("--exclude_frac", default=0.01, type=float)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    step_paper_isi_qc(
        derivatives_base=args.derivatives_base,
        out_csv_all=args.out_csv_all,
        out_csv_good=args.out_csv_good,
        threshold_ms=args.threshold_ms,
        exclude_frac=args.exclude_frac,
    )
