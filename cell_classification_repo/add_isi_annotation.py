import pandas as pd
from pathlib import Path

# Where your "noISI + min500" features live
base_feat = Path(r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir\hct_outputs")

feat_csv = base_feat / "unit_classification_features_noISI_min500spikes.csv"

# Where the QC file actually is (YOU said it's here)
qc_csv = Path(
    r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir\hct_qc_outputs\paper_qc_2ms_all_units.csv"
)

out_csv = base_feat / "unit_classification_features_noISI_min500spikes_withISI.csv"

print("Reading features:", feat_csv)
print("Reading QC:", qc_csv)

df_feat = pd.read_csv(feat_csv)
df_qc = pd.read_csv(qc_csv)

# force unit_id type match
df_feat["unit_id"] = pd.to_numeric(df_feat["unit_id"], errors="coerce").astype("Int64")
df_qc["unit_id"]   = pd.to_numeric(df_qc["unit_id"], errors="coerce").astype("Int64")

df_qc["meets_paper_isi"] = ~df_qc["paper_exclude"].astype(bool)

df_out = df_feat.merge(df_qc[["unit_id", "meets_paper_isi"]], on="unit_id", how="left")

df_out.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print("meets_paper_isi counts:\n", df_out["meets_paper_isi"].value_counts(dropna=False))
