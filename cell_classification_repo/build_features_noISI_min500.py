from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_FEATURES = ["peak_to_trough_ms", "firing_rate_hz", "burst_index", "oscillation_score"]


def step_build_features_noisi(
    feat4_csv: Path,
    out_csv_all: Path,
    out_csv_minspikes: Path,
    min_spikes: int = 500,
    features: list[str] = DEFAULT_FEATURES,
):
    """
    Build "no ISI filter" feature tables from the 4-metric table.

    Inputs:
      - feat4_csv: CSV containing (at least) unit_id, phy_group, n_spikes, and the 4 metrics.

    Outputs:
      - out_csv_all: all units, no ISI filtering
      - out_csv_minspikes: subset where n_spikes >= min_spikes (used for PCA/gating)

    Notes:
      - ISI QC is intentionally NOT used here.
    """
    feat4_csv = Path(feat4_csv)
    out_csv_all = Path(out_csv_all)
    out_csv_minspikes = Path(out_csv_minspikes)

    out_csv_all.parent.mkdir(parents=True, exist_ok=True)
    out_csv_minspikes.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feat4_csv)

    needed = ["unit_id", "phy_group", "n_spikes"] + list(features)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {feat4_csv}:\n  {missing}\nHave:\n  {list(df.columns)}"
        )

    df_out = df[needed].copy()

    # Save unfiltered (no ISI, all units)
    df_out.to_csv(out_csv_all, index=False)
    print("Saved (no ISI, all units):", out_csv_all)

    # Apply min spike filter
    df_min = df_out[df_out["n_spikes"] >= int(min_spikes)].copy()
    df_min.to_csv(out_csv_minspikes, index=False)

    print(f"Saved (no ISI, n_spikes>={min_spikes}):", out_csv_minspikes)
    print("Counts:", len(df_out), "->", len(df_min))


def parse_args():
    ap = argparse.ArgumentParser(description="Build no-ISI feature tables and a min-spikes subset.")
    ap.add_argument("--feat4_csv", required=True, type=Path)
    ap.add_argument("--out_csv_all", required=True, type=Path)
    ap.add_argument("--out_csv_minspikes", required=True, type=Path)
    ap.add_argument("--min_spikes", default=500, type=int)
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    step_build_features_noisi(
        feat4_csv=args.feat4_csv,
        out_csv_all=args.out_csv_all,
        out_csv_minspikes=args.out_csv_minspikes,
        min_spikes=args.min_spikes,
        features=args.features,
    )
