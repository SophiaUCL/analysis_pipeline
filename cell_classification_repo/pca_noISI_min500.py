from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


DEFAULT_FEATURES = ["peak_to_trough_ms", "firing_rate_hz", "burst_index", "oscillation_score"]


def step_pca(
    in_csv: Path,
    out_coords: Path,
    out_fig: Path,
    features: list[str] = DEFAULT_FEATURES,
    out_loadings_csv: Path | None = None,
    out_meta_json: Path | None = None,
):
    """
    PCA on `features` after z-scoring.
    Writes:
      - out_coords: unit_id, phy_group, n_spikes, PC1, PC2
      - out_fig: scatter of PC1 vs PC2
    Optionally writes:
      - out_loadings_csv: 2xF loadings (rows PC1/PC2, cols features)
      - out_meta_json: explained variance + basic info

    Notes:
      - Drops rows with NaN in any feature column.
      - Requires unit_id. Adds phy_group='unknown' if missing.
      - Keeps n_spikes if present; otherwise sets to NaN.
    """
    in_csv = Path(in_csv)
    out_coords = Path(out_coords)
    out_fig = Path(out_fig)
    if out_loadings_csv is not None:
        out_loadings_csv = Path(out_loadings_csv)
    if out_meta_json is not None:
        out_meta_json = Path(out_meta_json)

    out_coords.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    if out_loadings_csv is not None:
        out_loadings_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_meta_json is not None:
        out_meta_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    if "unit_id" not in df.columns:
        raise ValueError(f"{in_csv} must contain column 'unit_id'")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{in_csv} is missing PCA feature columns: {missing}")

    if "phy_group" not in df.columns:
        df["phy_group"] = "unknown"
    if "n_spikes" not in df.columns:
        df["n_spikes"] = np.nan

    df = df.dropna(subset=features).copy()

    X = df[features].to_numpy(float)
    Xz = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    PC = pca.fit_transform(Xz)

    df_out = df[["unit_id", "phy_group", "n_spikes"]].copy()
    df_out["PC1"] = PC[:, 0]
    df_out["PC2"] = PC[:, 1]
    df_out.to_csv(out_coords, index=False)

    # Save loadings + metadata (optional but recommended)
    explained = pca.explained_variance_ratio_.tolist()
    loadings = pd.DataFrame(pca.components_, columns=features, index=["PC1", "PC2"])

    if out_loadings_csv is not None:
        loadings.to_csv(out_loadings_csv, index=True)

    if out_meta_json is not None:
        meta = {
            "in_csv": str(in_csv),
            "n_rows_used": int(df_out.shape[0]),
            "features": features,
            "explained_variance_ratio": explained,
        }
        out_meta_json.write_text(json.dumps(meta, indent=2))

    # Plot
    plt.figure()
    plt.scatter(df_out["PC1"], df_out["PC2"], s=8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cell classification PCA (no ISI filter, min spikes subset)")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.close()

    print("Saved:", out_coords)
    print("Saved:", out_fig)
    print("Explained variance ratio:", explained)
    print("Loadings (rows PC1/PC2):")
    print(loadings)


# -----------------------
# CLI wrapper
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Run PCA on 4 cell-classification features.")
    ap.add_argument("--in_csv", required=True, type=Path)
    ap.add_argument("--out_coords", required=True, type=Path)
    ap.add_argument("--out_fig", required=True, type=Path)
    ap.add_argument("--out_loadings_csv", default=None, type=Path)
    ap.add_argument("--out_meta_json", default=None, type=Path)
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES,
                    help="Feature columns to use for PCA.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    step_pca(
        in_csv=args.in_csv,
        out_coords=args.out_coords,
        out_fig=args.out_fig,
        features=args.features,
        out_loadings_csv=args.out_loadings_csv,
        out_meta_json=args.out_meta_json,
    )
