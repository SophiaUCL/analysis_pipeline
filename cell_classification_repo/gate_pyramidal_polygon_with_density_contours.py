from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as MplPath
from scipy.stats import gaussian_kde


# -----------------------
# Density contour helper
# -----------------------
def add_kde_contours(ax, x, y, levels=(0.50, 0.70, 0.85, 0.93), gridsize=220):
    """
    Draw KDE contour guides.
    'levels' are quantiles over the grid density values (higher -> denser).
    """
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    xi = np.linspace(xmin, xmax, gridsize)
    yi = np.linspace(ymin, ymax, gridsize)
    xx, yy = np.meshgrid(xi, yi)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zflat = zz.ravel()
    zsorted = np.sort(zflat)
    thresh = [zsorted[int(q * (len(zsorted) - 1))] for q in levels]

    ax.contour(xx, yy, zz, levels=thresh)
    return thresh


def _load_pca(pca_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pca_csv)
    required = {"unit_id", "PC1", "PC2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pca_csv} is missing columns: {missing}")

    if "phy_group" not in df.columns:
        df["phy_group"] = "unknown"

    # ensure stable types
    df["unit_id"] = df["unit_id"].astype(str)
    df["phy_group"] = df["phy_group"].fillna("unknown").astype(str)
    df["PC1"] = df["PC1"].astype(float)
    df["PC2"] = df["PC2"].astype(float)
    return df


def _load_vertices(poly_csv: Path) -> np.ndarray:
    poly = pd.read_csv(poly_csv)
    if {"PC1", "PC2"}.issubset(poly.columns):
        verts = poly[["PC1", "PC2"]].to_numpy(float)
    elif {"x", "y"}.issubset(poly.columns):
        verts = poly[["x", "y"]].to_numpy(float)
    else:
        raise ValueError(f"Polygon CSV columns not recognized: {list(poly.columns)}")

    if len(verts) < 3:
        raise ValueError(f"Polygon needs >=3 vertices, got {len(verts)} in {poly_csv}")
    return verts


def _apply_polygon(df_pca: pd.DataFrame, verts: np.ndarray) -> pd.DataFrame:
    pts = df_pca[["PC1", "PC2"]].to_numpy(float)
    poly = MplPath(verts)
    inside = poly.contains_points(pts)

    out = df_pca.copy()
    out["cell_type"] = np.where(inside, "pyramidal", "interneuron")
    return out


def _plot_background(ax, df_pca: pd.DataFrame, show_legend: bool = True):
    x = df_pca["PC1"].to_numpy(float)
    y = df_pca["PC2"].to_numpy(float)

    groups = df_pca["phy_group"].fillna("unknown").astype(str)
    uniq = sorted(groups.unique())
    group_to_i = {g: i for i, g in enumerate(uniq)}
    c_int = groups.map(group_to_i).to_numpy(int)

    cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")

    ax.hexbin(x, y, gridsize=45, mincnt=1, linewidths=0.0, alpha=0.25)
    add_kde_contours(ax, x, y)
    ax.scatter(x, y, c=c_int, cmap=cmap, s=10, alpha=0.85)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if show_legend:
        # Legend handles (avoids scatter-empty bug)
        denom = max(1, len(uniq) - 1)
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=cmap(group_to_i[g] / denom),
                   markeredgecolor="none", markersize=7, label=g)
            for g in uniq
        ]
        ax.legend(handles=legend_handles, title="phy_group", loc="best", frameon=True)


def _save_overlay_png(df_pca: pd.DataFrame, verts: np.ndarray, overlay_png: Path):
    overlay_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    _plot_background(ax, df_pca, show_legend=True)

    # close polygon for plotting
    vx = np.r_[verts[:, 0], verts[0, 0]]
    vy = np.r_[verts[:, 1], verts[0, 1]]
    ax.plot(vx, vy, linewidth=2)

    ax.set_title("PCA (PC1/PC2) with saved polygon overlay")
    fig.savefig(overlay_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def step_polygon_gate(
    pca_csv: Path,
    labels_csv: Path,
    poly_csv: Path,
    overlay_png: Path | None = None,
    interactive: bool = False,
):
    """
    Contract:
    - If labels_csv exists: do nothing (assume already gated)
    - Else if poly_csv exists: apply polygon automatically and write labels_csv
    - Else if interactive: launch PolygonSelector, save poly_csv + labels_csv
    - Else: raise with a clear message

    Saves:
    - labels_csv: unit_id, cell_type, phy_group, PC1, PC2
    - poly_csv: vertices as columns PC1, PC2
    - overlay_png (optional): PCA plot with polygon overlay
    """
    pca_csv = Path(pca_csv)
    labels_csv = Path(labels_csv)
    poly_csv = Path(poly_csv)
    if overlay_png is not None:
        overlay_png = Path(overlay_png)

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    poly_csv.parent.mkdir(parents=True, exist_ok=True)

    if labels_csv.exists():
        print(f"✅ Labels already exist, skipping gating:\n  {labels_csv}")
        return

    df = _load_pca(pca_csv)

    # --- Non-interactive: apply saved polygon if present
    if poly_csv.exists() and not interactive:
        verts = _load_vertices(poly_csv)
        df_out = _apply_polygon(df, verts)
        df_out[["unit_id", "cell_type", "phy_group", "PC1", "PC2"]].to_csv(labels_csv, index=False)

        n_pyr = int((df_out["cell_type"] == "pyramidal").sum())
        n_int = int((df_out["cell_type"] == "interneuron").sum())
        print(f"Saved: {labels_csv}")
        print(f"Used polygon: {poly_csv}")
        print(f"Pyramidal: {n_pyr} | Interneuron: {n_int} | Total: {len(df_out)}")

        if overlay_png is not None:
            _save_overlay_png(df, verts, overlay_png)
            print(f"Saved overlay: {overlay_png}")
        return

    # --- Interactive required if no polygon exists
    if not interactive and not poly_csv.exists():
        raise RuntimeError(
            f"No saved polygon found at:\n  {poly_csv}\n"
            f"Run with --interactive to draw one (or provide an existing polygon file)."
        )

    # --- Interactive gating
    x = df["PC1"].to_numpy(float)
    y = df["PC2"].to_numpy(float)
    pts = np.column_stack([x, y])

    fig, ax = plt.subplots()
    _plot_background(ax, df, show_legend=True)

    ax.set_title(
        "Draw polygon around PYRAMIDAL cluster.\n"
        "Use density contours as a guide.\n"
        "Click vertices, double-click (or Enter) to finish."
    )

    selector = {"obj": None}  # mutable holder so we can disconnect inside callback

    def onselect(verts_list):
        verts = np.asarray(verts_list, dtype=float)
        if verts.shape[0] < 3:
            print("Polygon needs >=3 vertices; ignoring selection.")
            return

        poly = MplPath(verts)
        inside = poly.contains_points(pts)

        df_out = df.copy()
        df_out["cell_type"] = np.where(inside, "pyramidal", "interneuron")

        df_out[["unit_id", "cell_type", "phy_group", "PC1", "PC2"]].to_csv(labels_csv, index=False)
        pd.DataFrame(verts, columns=["PC1", "PC2"]).to_csv(poly_csv, index=False)

        n_pyr = int(inside.sum())
        n_int = int((~inside).sum())
        print(f"\nSaved: {labels_csv}")
        print(f"Saved polygon vertices: {poly_csv}")
        print(f"Pyramidal: {n_pyr} | Interneuron: {n_int} | Total: {len(df_out)}")

        if overlay_png is not None:
            _save_overlay_png(df, verts, overlay_png)
            print(f"Saved overlay: {overlay_png}")

        # Visual feedback
        ax.clear()
        ax.hexbin(x, y, gridsize=45, mincnt=1, linewidths=0.0, alpha=0.25)
        add_kde_contours(ax, x, y)
        ax.scatter(x[inside], y[inside], s=10, alpha=0.9)
        ax.scatter(x[~inside], y[~inside], s=10, alpha=0.25)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"Saved. Pyramidal={n_pyr}, Interneuron={n_int}\nClose window to finish.")
        fig.canvas.draw_idle()

        if selector["obj"] is not None:
            selector["obj"].disconnect_events()

    selector["obj"] = PolygonSelector(ax, onselect, useblit=True)
    plt.show()


# -----------------------
# CLI wrapper
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Polygon gating for pyramidal vs interneuron on PCA space.")
    ap.add_argument("--pca_csv", required=True, type=Path)
    ap.add_argument("--labels_csv", required=True, type=Path)
    ap.add_argument("--poly_csv", required=True, type=Path)
    ap.add_argument("--overlay_png", default=None, type=Path,
                    help="Optional: save a PCA plot with polygon overlay (if vertices exist / after drawing).")
    ap.add_argument("--interactive", action="store_true",
                    help="Launch interactive polygon drawing if needed.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    step_polygon_gate(
        pca_csv=args.pca_csv,
        labels_csv=args.labels_csv,
        poly_csv=args.poly_csv,
        overlay_png=args.overlay_png,
        interactive=args.interactive,
    )
