import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---- paths ----
base_dir = Path(r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir\hct_outputs")
in_csv = base_dir / "unit_classification_features.csv"

out_pca_csv = base_dir / "unit_classification_pca_coords_paper_included.csv"
out_fig = base_dir / "unit_classification_PC1_PC2.png"

# ---- load ----
df = pd.read_csv(in_csv)

# ---- filter to paper-included ----
df = df[df["meets_paper_isi"] == True].copy()

# ---- select features ----
features = ["peak_to_trough_ms", "firing_rate_hz", "burst_index", "oscillation_score"]

# drop rows missing any feature
df = df.dropna(subset=features).copy()

X = df[features].to_numpy(dtype=float)

# ---- standardize (z-score) ----
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# ---- PCA ----
pca = PCA(n_components=2)
PC = pca.fit_transform(Xz)

df_out = df[["unit_id", "phy_group", "meets_paper_isi"]].copy()
df_out["PC1"] = PC[:, 0]
df_out["PC2"] = PC[:, 1]

# ---- save coords ----
df_out.to_csv(out_pca_csv, index=False)

print("Saved PCA coords:", out_pca_csv)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("PC loadings (rows=PCs, cols=features):\n", pca.components_)

# ---- plot ----
plt.figure()
plt.scatter(df_out["PC1"], df_out["PC2"], s=8)  # no styling yet (paper-like)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("HCT unit classification PCA (paper-included units)")
plt.tight_layout()
plt.savefig(out_fig, dpi=200)
plt.show()

print("Saved figure:", out_fig)
