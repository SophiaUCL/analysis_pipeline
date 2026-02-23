import pandas as pd
from pathlib import Path

from make_unit_features_4metrics_good_and_all_units import load_phy_groups


# =========================
# EDIT THESE PATHS ONLY
# =========================

subject = "sub-002_id-1R"
session = "ses-02_date-11092025"

# Local output (safe while Ceph is full)
out_dir = Path(
    r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir\hct_outputs"
)
out_dir.mkdir(parents=True, exist_ok=True)

# Base path for Phy (must contain cluster_group.tsv somewhere underneath)
derivatives_base = (
    r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task"
    fr"\derivatives\{subject}\{session}\all_trials"
)

# --- INPUT CSVs ---

# Paper ISI QC (ALL units, computed locally)
paper_qc_csv = Path(
    r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir"
    r"\hct_qc_outputs\paper_qc_2ms_all_units.csv"
)

# Burst index (ALL units)
burst_csv = Path(
    r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\cell_classification_projects\HCT"
    fr"\{subject}\{session}\audits\burst_index\burst_metrics_all_units.csv"
)

# 4-metric features (ALL units)
feat4_csv = Path(
    r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\cell_classification_projects\HCT"
    fr"\{subject}\{session}\outputs\unit_features\unit_features_4metrics_all_units.csv"
)


# =========================
# LOAD
# =========================

for p in [paper_qc_csv, burst_csv, feat4_csv]:
    if not p.exists():
        raise FileNotFoundError(f"Missing input file:\n{p}")

df_phy   = load_phy_groups(derivatives_base)   # unit_id, phy_group
df_qc    = pd.read_csv(paper_qc_csv)
df_burst = pd.read_csv(burst_csv)
df_feat  = pd.read_csv(feat4_csv)


# =========================
# UNIT ID ALIGNMENT
# =========================

for df in (df_phy, df_qc, df_burst, df_feat):
    df["unit_id"] = pd.to_numeric(df["unit_id"], errors="coerce").astype("Int64")


# =========================
# PAPER ISI FLAG
# =========================

if "paper_exclude" not in df_qc.columns:
    raise ValueError(
        f"'paper_exclude' not found in {paper_qc_csv.name}. "
        f"Columns are: {list(df_qc.columns)}"
    )

df_qc["meets_paper_isi"] = ~df_qc["paper_exclude"].astype(bool)


# =========================
# MERGE (NO RECOMPUTE)
# =========================

final = (
    df_phy
    .merge(df_qc[["unit_id", "meets_paper_isi"]], on="unit_id", how="left")
    .merge(
        df_feat[
            ["unit_id", "firing_rate_hz", "peak_to_trough_ms", "oscillation_score"]
        ],
        on="unit_id",
        how="left",
    )
    .merge(df_burst[["unit_id", "burst_index"]], on="unit_id", how="left")
)

final = final[
    [
        "unit_id",
        "meets_paper_isi",
        "phy_group",
        "firing_rate_hz",
        "peak_to_trough_ms",
        "burst_index",
        "oscillation_score",
    ]
].copy()


# =========================
# SANITY CHECKS
# =========================

print("\nTotal units:", len(final))
print("\nMissing fraction by column:")
print(final.isna().mean().sort_values(ascending=False))

print("\nPhy groups:")
print(final["phy_group"].value_counts(dropna=False))

print("\nPaper ISI pass/fail:")
print(final["meets_paper_isi"].value_counts(dropna=False))


# =========================
# SAVE
# =========================

out_csv = out_dir / "unit_classification_features.csv"
final.to_csv(out_csv, index=False)

# =========================
# PAPER-INCLUDED-ONLY CSV
# =========================

included_csv = out_dir / "unit_classification_features_paper_included.csv"

final_included = final[final["meets_paper_isi"] == True].copy()

final_included.to_csv(included_csv, index=False)

print(
    f"\nSaved paper-included-only CSV: {included_csv}\n"
    f"Included units: {len(final_included)} / {len(final)}"
)


print("\nSaved:", out_csv)
