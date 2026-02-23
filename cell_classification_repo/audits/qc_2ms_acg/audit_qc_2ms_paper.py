from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import spikeinterface as si

def choose_out_dir(ceph_dir: Path, local_fallback: Path) -> Path:
    """
    Try to use ceph_dir; if it's not writable (quota/full), fall back to local_fallback.
    Cleans up the test file on success.
    """
    ceph_dir = Path(ceph_dir)
    local_fallback = Path(local_fallback)

    try:
        ceph_dir.mkdir(parents=True, exist_ok=True)
        test_path = ceph_dir / "_write_test.tmp"
        with open(test_path, "w") as f:
            f.write("ok")
        test_path.unlink(missing_ok=True)
        print("Writing outputs to Ceph:", ceph_dir)
        return ceph_dir
    except OSError as e:
        print("\nWARNING: Ceph output not writable, falling back to local.")
        print("Ceph dir:", ceph_dir)
        print("Reason:", repr(e))
        local_fallback.mkdir(parents=True, exist_ok=True)
        print("Writing outputs locally:", local_fallback, "\n")
        return local_fallback


def _safe_out_dir(primary: Path) -> Path:
    """
    Use primary out_dir if writable; otherwise fall back to local.
    """
    try:
        primary.mkdir(parents=True, exist_ok=True)
        test = primary / "_write_test.tmp"
        test.write_text("ok")
        test.unlink()
        return primary
    except OSError as e:
        print("\nWARNING: cannot write to", primary)
        print("Reason:", repr(e))
        fallback = Path.home() / "Desktop" / "hct_qc_outputs"
        fallback.mkdir(parents=True, exist_ok=True)
        print("Falling back to local:", fallback, "\n")
        return fallback

def _get_fs_hz_from_analyzer(sorting_analyzer) -> float:
    """Try to read sampling frequency from the loaded SpikeInterface analyzer."""
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
    if "cluster_id" not in df.columns or "group" not in df.columns:
        raise ValueError(f"Unexpected columns in cluster_group.tsv: {list(df.columns)}")

    good = df.loc[df["group"] == "good", "cluster_id"].to_numpy(dtype=int)
    return good


def frac_spikes_with_next_isi_below(
    spike_times_s: np.ndarray, threshold_ms: float = 2.0
) -> Tuple[float, int, int]:
    """
    Fraction of spikes whose NEXT inter-spike interval is below threshold.
    Returns (fraction, n_violations, n_spikes).

    This is a standard operationalization of "refractory violations".
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


def audit_paper_qc_2ms(
    derivatives_base: str,
    project_base: str,
    threshold_ms: float = 2.0,
    exclude_frac: float = 0.01,
) -> Path:
    """
    Paper-style QC audit:
    For Phy-curated GOOD units, compute fraction of spikes with next-ISI < 2 ms.
    Flag units with fraction > 1%.

    READS from derivatives_base only; WRITES to project_base/audits/qc_2ms_acg
    """
    # ---- INPUT (read only) ----
    unit_features_path = os.path.join(
        derivatives_base, "analysis", "cell_characteristics", "unit_features"
    )
    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")

    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Missing sorting_analyzer:\n{analyzer_path}")

    good_units = _load_good_units_from_cluster_group(derivatives_base)

    print("Loading sorting_analyzer (read-only):")
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

    # ---- OUTPUT (write to Chiara folder) ----
    ceph_out_dir = Path(project_base) / "audits" / "qc_2ms_acg"
    local_out_dir = Path(r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir\hct_qc_outputs")

    out_dir = choose_out_dir(ceph_out_dir, local_out_dir)

    def _compute_qc_df(unit_ids: np.ndarray) -> pd.DataFrame:
        rows = []
        for uid in unit_ids:
            st_samples = sorting.get_unit_spike_train(unit_id=int(uid))
            st_s = np.asarray(st_samples, dtype=float) / fs_hz

            frac, n_viol, n_spk = frac_spikes_with_next_isi_below(
                st_s, threshold_ms=threshold_ms
            )
            rows.append(
                dict(
                    unit_id=int(uid),
                    n_spikes=int(n_spk),
                    n_nextISI_lt_threshold=int(n_viol),
                    frac_nextISI_lt_threshold=frac,
                    threshold_ms=float(threshold_ms),
                    paper_exclude=bool(np.isfinite(frac) and frac > exclude_frac),
                    exclude_frac=float(exclude_frac),
                )
            )

        return pd.DataFrame(rows).sort_values(
            "frac_nextISI_lt_threshold", ascending=False
        )

    # ---- Compute + save GOOD units QC ----
    df_good = _compute_qc_df(good_units)
    out_csv_good = out_dir / "paper_qc_2ms_good_units.csv"
    df_good.to_csv(out_csv_good, index=False)

    print("\nSaved paper-style QC audit (GOOD units) to:")
    print(" ", out_csv_good)

    n_excl_good = int(df_good["paper_exclude"].sum())
    print(
        f"\nGood units: {len(df_good)} | Paper-excluded (>{exclude_frac*100:.1f}% under {threshold_ms} ms): {n_excl_good}"
    )

    if n_excl_good > 0:
        print("\nTop 10 worst offenders (GOOD units):")
        print(
            df_good.head(10)[
                [
                    "unit_id",
                    "n_spikes",
                    "n_nextISI_lt_threshold",
                    "frac_nextISI_lt_threshold",
                    "paper_exclude",
                ]
            ]
        )

    # ---- Compute + save ALL units QC ----
    df_all = _compute_qc_df(all_units)
    out_csv_all = out_dir / "paper_qc_2ms_all_units.csv"
    df_all.to_csv(out_csv_all, index=False)

    print("\nSaved paper-style QC audit (ALL units) to:")
    print(" ", out_csv_all)

    n_excl_all = int(df_all["paper_exclude"].sum())
    print(
        f"\nAll units: {len(df_all)} | Paper-excluded (>{exclude_frac*100:.1f}% under {threshold_ms} ms): {n_excl_all}"
    )

    if n_excl_all > 0:
        print("\nTop 10 worst offenders (ALL units):")
        print(
            df_all.head(10)[
                [
                    "unit_id",
                    "n_spikes",
                    "n_nextISI_lt_threshold",
                    "frac_nextISI_lt_threshold",
                    "paper_exclude",
                ]
            ]
        )

    return out_csv_all


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

    audit_paper_qc_2ms(
        derivatives_base=derivatives_base,
        project_base=project_base,
        threshold_ms=2.0,
        exclude_frac=0.01,
    )
