#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import pandas as pd


# -------------------------
# IMPORT PIPELINE STEPS
# -------------------------
from cell_classification_repo.make_unit_features_4metrics_good_and_all_units import step_metrics
from cell_classification_repo.paper_qc_2ms import step_paper_isi_qc
from cell_classification_repo.build_features_noISI_min500 import step_build_features_noisi
from cell_classification_repo.pca_noISI_min500 import step_pca
from cell_classification_repo.gate_pyramidal_polygon_with_density_contours import step_polygon_gate


# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Hippocampal cell classification pipeline (pyramidal vs interneuron)"
    )
    ap.add_argument("--derivatives_base", required=True, type=Path)

    # Optional override (e.g. Ceph full, local testing)
    ap.add_argument("--out_base", required=False, type=Path, default=None)

    ap.add_argument("--min_spikes", default=500, type=int)

    ap.add_argument(
        "--interactive_gate",
        action="store_true",
        help="Launch interactive polygon gating if no polygon exists.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute steps even if outputs already exist.",
    )
    ap.add_argument(
        "--stop_after",
        default=None,
        choices=["metrics", "isi_qc", "build_features", "pca", "gate", "merge", "inspection"],
    )

    # ISI QC knobs (paper defaults)
    ap.add_argument("--isi_threshold_ms", default=2.0, type=float)
    ap.add_argument("--isi_exclude_frac", default=0.01, type=float)

    # Inspection knobs
    ap.add_argument(
        "--inspect_scope",
        default="labeled",
        choices=["labeled", "all"],
        help="Which units to render inspection sheets for. "
             "'labeled' = only units that went through PCA/gating (recommended). "
             "'all' = all units in final table.",
    )
    ap.add_argument(
        "--inspect_format",
        default="png",
        choices=["png", "pdf"],
        help="Output format for inspection sheets (one file per unit).",
    )

    return ap.parse_args()


# -------------------------
# Canonical paths
# -------------------------
@dataclass(frozen=True)
class Paths:
    derivatives_base: Path
    out_base: Path

    @property
    def root(self) -> Path:
        return (
            self.out_base
            / "analysis"
            / "cell_characteristics"
            / "unit_features"
            / "cell_classification"
        )

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def pca(self) -> Path:
        return self.root / "pca"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def inspection(self) -> Path:
        return self.root / "inspection_sheets"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure_dirs(self):
        for p in [self.tables, self.pca, self.figures, self.inspection, self.logs]:
            p.mkdir(parents=True, exist_ok=True)

    # ---- filenames
    @property
    def csv_metrics_all(self) -> Path:
        return self.tables / "unit_metrics_4features_all_units.csv"

    @property
    def csv_metrics_good(self) -> Path:
        return self.tables / "unit_metrics_4features_good_units.csv"

    @property
    def csv_isi_all(self) -> Path:
        return self.tables / "paper_isi_qc_2ms_all_units.csv"

    @property
    def csv_isi_good(self) -> Path:
        return self.tables / "paper_isi_qc_2ms_good_units.csv"

    @property
    def csv_noisi_all(self) -> Path:
        return self.tables / "unit_features_noISI_all_units.csv"

    @property
    def csv_noisi_minspikes(self) -> Path:
        return self.tables / "unit_features_noISI_minspikes.csv"

    @property
    def csv_pca_coords(self) -> Path:
        return self.pca / "pca_coords_noISI_minspikes.csv"

    @property
    def fig_pca(self) -> Path:
        return self.pca / "pca_PC1_PC2_noISI_minspikes.png"

    @property
    def pca_loadings_csv(self) -> Path:
        return self.pca / "pca_loadings_PC1_PC2.csv"

    @property
    def pca_meta_json(self) -> Path:
        return self.pca / "pca_meta.json"

    @property
    def csv_polygon_vertices(self) -> Path:
        return self.pca / "polygon_vertices_PC1_PC2.csv"

    @property
    def csv_polygon_labels(self) -> Path:
        return self.pca / "labels_polygon_noISI_minspikes.csv"

    @property
    def pca_overlay_png(self) -> Path:
        return self.pca / "pca_cluster_with_polygon_overlay.png"

    @property
    def csv_final_features(self) -> Path:
        return self.tables / "unit_classification_features_final.csv"

    @property
    def run_manifest(self) -> Path:
        return self.logs / "run_manifest.json"


# -------------------------
# Helpers
# -------------------------
def should_run(out_path: Path, force: bool) -> bool:
    return force or (not out_path.exists())


def write_manifest(paths: Paths, args: argparse.Namespace):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "derivatives_base": str(paths.derivatives_base),
        "out_base": str(paths.out_base),
        "pipeline_root": str(paths.root),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    paths.run_manifest.write_text(json.dumps(payload, indent=2))


# -------------------------
# Final merge step
# -------------------------
def step_merge_final(
    noisi_all_csv: Path,
    isi_qc_all_csv: Path,
    polygon_labels_csv: Path,
    out_final_csv: Path,
):
    df_feat = pd.read_csv(noisi_all_csv)
    df_isi = pd.read_csv(isi_qc_all_csv)
    df_lab = pd.read_csv(polygon_labels_csv)

    for df in (df_feat, df_isi, df_lab):
        df["unit_id"] = pd.to_numeric(df["unit_id"], errors="coerce").astype("Int64")

    if "meets_paper_isi" not in df_isi.columns:
        df_isi["meets_paper_isi"] = ~df_isi["paper_exclude"].astype(bool)

    df_isi = df_isi[["unit_id", "meets_paper_isi"]]
    df_lab = df_lab[["unit_id", "cell_type"]]

    final = (
        df_feat
        .merge(df_isi, on="unit_id", how="left")
        .merge(df_lab, on="unit_id", how="left")
    )

    final["has_polygon_label"] = final["cell_type"].notna()

    out_final_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_final_csv, index=False)

    print("\nSaved final classification table:")
    print(" ", out_final_csv)
    print("Total units:", len(final))


# -------------------------
# Inspection sheets
# -------------------------
def _find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def step_inspection(
    derivatives_base: Path,
    final_features_csv: Path,
    out_dir: Path,
    scope: str = "labeled",
    out_format: str = "png",
):
    """
    Assemble inspection sheets from existing images and annotate with unit metadata.

    - Writes one file per unit to out_dir.
    - Does NOT recompute waveforms/ACGs/ratemaps; it only loads images if present.

    Expected (but flexible) image locations under derivatives_base:
      - Waveforms: analysis/cell_characteristics/unit_features/**/waveform* or *waveforms*
      - ACGs:      analysis/cell_characteristics/unit_features/**/acg* or *acgs*
      - Ratemaps:  analysis/**/ratemap* (session/goal1/goal2 variants supported)

    If your real filenames differ, adjust the candidate filename patterns below.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    derivatives_base = Path(derivatives_base)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(final_features_csv)

    if scope == "labeled":
        if "has_polygon_label" not in df.columns:
            df["has_polygon_label"] = df.get("cell_type").notna()
        df = df[df["has_polygon_label"]].copy()

    print(f"Inspection: rendering {len(df)} units (scope={scope}) -> {out_dir}")

    # Candidate roots (search is cheap; these are stable anchors)
    analysis_dir = derivatives_base / "analysis"
    unit_feat_dir = analysis_dir / "cell_characteristics" / "unit_features"

    # Some common subfolders people use (adjust if needed)
    wf_roots = [
        unit_feat_dir / "waveforms",
        unit_feat_dir / "all_units_overview",
        unit_feat_dir / "figures",
        unit_feat_dir,
    ]
    acg_roots = [
        unit_feat_dir / "acgs",
        unit_feat_dir / "all_units_overview",
        unit_feat_dir / "figures",
        unit_feat_dir,
    ]
    rm_roots = [
        analysis_dir / "cell_characteristics" / "spatial_features",
        analysis_dir / "cell_characteristics" / "spatial_features" / "ratemaps",
        analysis_dir,
    ]

    # Helper: candidate filenames for a given unit id
    def waveform_candidates(uid: int) -> list[Path]:
        names = [
            f"unit_{uid}_waveform.png",
            f"{uid}_waveform.png",
            f"waveform_unit_{uid}.png",
            f"waveform_{uid}.png",
        ]
        out = []
        for r in wf_roots:
            for n in names:
                out.append(r / n)
        return out

    def acg_candidates(uid: int) -> list[Path]:
        names = [
            f"unit_{uid}_acg.png",
            f"{uid}_acg.png",
            f"acg_unit_{uid}.png",
            f"acg_{uid}.png",
            f"autocorr_unit_{uid}.png",
            f"autocorr_{uid}.png",
        ]
        out = []
        for r in acg_roots:
            for n in names:
                out.append(r / n)
        return out

    def ratemap_candidates(uid: int) -> list[Path]:
        # session/goal variants supported
        names = [
            f"unit_{uid}_ratemap.png",
            f"{uid}_ratemap.png",
            f"ratemap_unit_{uid}.png",
            f"ratemap_{uid}.png",
            f"unit_{uid}_ratemap_session.png",
            f"unit_{uid}_ratemap_goal1.png",
            f"unit_{uid}_ratemap_goal2.png",
        ]
        out = []
        for r in rm_roots:
            for n in names:
                out.append(r / n)
        return out

    # Render
    for _, row in df.iterrows():
        uid = int(row["unit_id"])

        wf_img = _find_first_existing(waveform_candidates(uid))
        acg_img = _find_first_existing(acg_candidates(uid))
        rm_img = _find_first_existing(ratemap_candidates(uid))

        imgs = []
        titles = []
        for p, t in [(wf_img, "Waveform"), (acg_img, "ACG"), (rm_img, "Ratemap")]:
            if p is not None and p.exists():
                imgs.append(Image.open(p))
                titles.append(f"{t}\n{p.name}")

        if not imgs:
            # nothing to render for this unit
            continue

        fig, axes = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 4))
        if len(imgs) == 1:
            axes = [axes]

        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=9)

        # Safe formatting for possibly-missing numeric values
        def fmt(x, nd=2):
            try:
                if pd.isna(x):
                    return "NA"
                return f"{float(x):.{nd}f}"
            except Exception:
                return str(x)

        annot = (
            f"Unit {uid} | cell_type={row.get('cell_type', 'NA')} | "
            f"meets_paper_isi={row.get('meets_paper_isi', 'NA')} | "
            f"n_spikes={row.get('n_spikes', 'NA')}\n"
            f"FR={fmt(row.get('firing_rate_hz'))} Hz | "
            f"PT={fmt(row.get('peak_to_trough_ms'))} ms | "
            f"BI={fmt(row.get('burst_index'))} | "
            f"Osc={fmt(row.get('oscillation_score'))}"
        )
        fig.suptitle(annot, fontsize=10)
        fig.tight_layout()

        if out_format == "png":
            out_path = out_dir / f"unit_{uid:04d}_inspection.png"
            fig.savefig(out_path, dpi=200)
        else:
            out_path = out_dir / f"unit_{uid:04d}_inspection.pdf"
            fig.savefig(out_path)

        plt.close(fig)

    print("✅ Inspection sheets complete.")
    print("Output dir:", out_dir)


def run_pipeline(
    derivatives_base: Path,
    out_base: Path | None = None,
    min_spikes: int = 500,
    interactive_gate: bool =True,
    force: bool = False,
    stop_after: str | None = None,
    isi_threshold_ms: float = 2.0,
    isi_exclude_frac: float = 0.01,
    inspect_scope: str = "labeled",
    inspect_format: str = "png",
):
    out_base = out_base if out_base is not None else derivatives_base

    args = argparse.Namespace(
        derivatives_base=Path(derivatives_base),
        out_base=Path(out_base) if out_base is not None else None,
        min_spikes=min_spikes,
        interactive_gate=interactive_gate,
        force=force,
        stop_after=stop_after,
        isi_threshold_ms=isi_threshold_ms,
        isi_exclude_frac=isi_exclude_frac,
        inspect_scope=inspect_scope,
        inspect_format=inspect_format,
    )

    paths = Paths(
        derivatives_base=args.derivatives_base,
        out_base=args.out_base if args.out_base is not None else args.derivatives_base,
    )
    paths.ensure_dirs()
    write_manifest(paths, args)

    # ---- identical logic to main() from here ----
    if should_run(paths.csv_metrics_all, args.force):
        step_metrics(
            derivatives_base=str(paths.derivatives_base),
            out_csv_all=paths.csv_metrics_all,
            out_csv_good=paths.csv_metrics_good,
        )
    if args.stop_after == "metrics":
        return paths

    if should_run(paths.csv_isi_all, args.force):
        step_paper_isi_qc(
            derivatives_base=str(paths.derivatives_base),
            out_csv_all=paths.csv_isi_all,
            out_csv_good=paths.csv_isi_good,
            threshold_ms=args.isi_threshold_ms,
            exclude_frac=args.isi_exclude_frac,
        )
    if args.stop_after == "isi_qc":
        return paths

    if should_run(paths.csv_noisi_all, args.force):
        step_build_features_noisi(
            feat4_csv=paths.csv_metrics_all,
            out_csv_all=paths.csv_noisi_all,
            out_csv_minspikes=paths.csv_noisi_minspikes,
            min_spikes=args.min_spikes,
        )
    if args.stop_after == "build_features":
        return paths

    if should_run(paths.csv_pca_coords, args.force):
        step_pca(
            in_csv=paths.csv_noisi_minspikes,
            out_coords=paths.csv_pca_coords,
            out_fig=paths.fig_pca,
            out_loadings_csv=paths.pca_loadings_csv,
            out_meta_json=paths.pca_meta_json,
        )
    if args.stop_after == "pca":
        return paths

    step_polygon_gate(
        pca_csv=paths.csv_pca_coords,
        labels_csv=paths.csv_polygon_labels,
        poly_csv=paths.csv_polygon_vertices,
        overlay_png=paths.pca_overlay_png,
        interactive=args.interactive_gate,
    )
    if args.stop_after == "gate":
        return paths

    if should_run(paths.csv_final_features, args.force):
        step_merge_final(
            noisi_all_csv=paths.csv_noisi_all,
            isi_qc_all_csv=paths.csv_isi_all,
            polygon_labels_csv=paths.csv_polygon_labels,
            out_final_csv=paths.csv_final_features,
        )
    if args.stop_after == "merge":
        return paths

    step_inspection(
        derivatives_base=paths.derivatives_base,
        final_features_csv=paths.csv_final_features,
        out_dir=paths.inspection,
        scope=args.inspect_scope,
        out_format=args.inspect_format,
    )

    return paths


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()

    # ✅ DEFAULT: write outputs inside derivatives_base
    out_base = args.out_base if args.out_base is not None else args.derivatives_base

    paths = Paths(
        derivatives_base=args.derivatives_base,
        out_base=out_base,
    )
    paths.ensure_dirs()
    write_manifest(paths, args)
    print("Step 1")
    # 1) metrics
    if should_run(paths.csv_metrics_all, args.force):
        step_metrics(
            derivatives_base=str(paths.derivatives_base),
            out_csv_all=paths.csv_metrics_all,
            out_csv_good=paths.csv_metrics_good,
        )
    if args.stop_after == "metrics":
        return
    print("Step 2")
    # 2) ISI QC
    if should_run(paths.csv_isi_all, args.force):
        step_paper_isi_qc(
            derivatives_base=str(paths.derivatives_base),
            out_csv_all=paths.csv_isi_all,
            out_csv_good=paths.csv_isi_good,
            threshold_ms=args.isi_threshold_ms,
            exclude_frac=args.isi_exclude_frac,
        )
    if args.stop_after == "isi_qc":
        return

    print("Step 3")
    # 3) build no-ISI feature tables
    if should_run(paths.csv_noisi_all, args.force):
        step_build_features_noisi(
            feat4_csv=paths.csv_metrics_all,
            out_csv_all=paths.csv_noisi_all,
            out_csv_minspikes=paths.csv_noisi_minspikes,
            min_spikes=args.min_spikes,
        )
    if args.stop_after == "build_features":
        return

    print("Step 4")
    # 4) PCA
    if should_run(paths.csv_pca_coords, args.force):
        step_pca(
            in_csv=paths.csv_noisi_minspikes,
            out_coords=paths.csv_pca_coords,
            out_fig=paths.fig_pca,
            out_loadings_csv=paths.pca_loadings_csv,
            out_meta_json=paths.pca_meta_json,
        )
    if args.stop_after == "pca":
        return

    print("Step 5")
    # 5) Polygon gating (resumable)
    step_polygon_gate(
        pca_csv=paths.csv_pca_coords,
        labels_csv=paths.csv_polygon_labels,
        poly_csv=paths.csv_polygon_vertices,
        overlay_png=paths.pca_overlay_png,
        interactive=args.interactive_gate,
    )
    if args.stop_after == "gate":
        return

    print("Step 6")
    # 6) Final merge
    if should_run(paths.csv_final_features, args.force):
        step_merge_final(
            noisi_all_csv=paths.csv_noisi_all,
            isi_qc_all_csv=paths.csv_isi_all,
            polygon_labels_csv=paths.csv_polygon_labels,
            out_final_csv=paths.csv_final_features,
        )
    if args.stop_after == "merge":
        return

    print("Step 7")
    # 7) Inspection sheets (assemble images + annotate)
    # Note: this step does not depend on "force" by default because it's often safe to re-render.
    step_inspection(
        derivatives_base=paths.derivatives_base,
        final_features_csv=paths.csv_final_features,
        out_dir=paths.inspection,
        scope=args.inspect_scope,
        out_format=args.inspect_format,
    )
    if args.stop_after == "inspection":
        return

    print(f"\n✅ Pipeline complete.\nFinal table:\n  {paths.csv_final_features}\n")
    print(f"Inspection sheets:\n  {paths.inspection}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {type(e).__name__}: {e}\n", file=sys.stderr)
        raise
