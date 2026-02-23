import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
base = Path(r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir\hct_outputs")

feat_csv = base / "unit_classification_features_noISI_min500spikes.csv"
lab_csv  = base / "labels_polygon_noISI_min500.csv"

fig_dir = base / "sanity_check_figures_noISI_min500"
fig_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# Load + merge
# -----------------------
df = pd.read_csv(feat_csv)
lab = pd.read_csv(lab_csv)[["unit_id", "cell_type"]]

df = df.merge(lab, on="unit_id", how="inner")

metrics = [
    "peak_to_trough_ms",
    "firing_rate_hz",
    "burst_index",
    "oscillation_score",
]

print("\nCell counts:")
print(df["cell_type"].value_counts())

# -----------------------
# Summary stats
# -----------------------
summary = df.groupby("cell_type")[metrics].agg(["count", "median", "mean"])
print("\nSummary (count / median / mean):\n")
print(summary)

summary.to_csv(fig_dir / "metric_summary_by_cell_type.csv")

# -----------------------
# Histograms (saved)
# -----------------------
for m in metrics:
    plt.figure(figsize=(5, 4))

    for ct in ["pyramidal", "interneuron"]:
        s = df.loc[df["cell_type"] == ct, m].dropna()
        plt.hist(
            s,
            bins=40,
            alpha=0.6,
            density=True,
            label=f"{ct} (n={len(s)})",
        )

    plt.xlabel(m)
    plt.ylabel("density")
    plt.title(m.replace("_", " "))
    plt.legend()
    plt.tight_layout()

    out_png = fig_dir / f"{m}_pyramidal_vs_interneuron.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Saved:", out_png)

print("\nAll figures saved to:", fig_dir)
