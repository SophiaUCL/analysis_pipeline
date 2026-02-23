import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# =========================
# EDIT THESE PATHS
# =========================

base_local = Path(r"C:\Users\Okeefe Lab\Desktop\data analysis\cell_classif_dir\hct_outputs")

# Features CSV (with meets_paper_isi added)
features_csv = base_local / "unit_classification_features_noISI_min500spikes_withISI.csv"

# Polygon labels from PCA gate (contains unit_id, cell_type)
labels_csv = base_local / "labels_polygon_noISI_min500.csv"

# Ceph folders with existing images
auto_wv_dir = Path(
    r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task"
    r"\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    r"\analysis\cell_characteristics\unit_features\auto_and_wv"
)

ratemap_dir = Path(
    r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task"
    r"\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    r"\analysis\cell_characteristics\spatial_features\ratemaps_and_hd"
)

# Where to save the inspection sheets (local)
out_dir = base_local / "inspection_sheets_noISI_min500_ALL"
out_dir.mkdir(parents=True, exist_ok=True)

# Optional: restrict to a few units for testing
MANUAL_UNIT_IDS = []  # e.g. [1, 18, 59]


# =========================
# Helpers
# =========================

def find_image_for_unit(folder: Path, unit_id: int, prefer_contains: str | None = None):
    """
    Robust to zero padding:
      matches unit_1, unit_001, unit-01, unit0001, etc.
    Works by parsing the integer after 'unit' and comparing numerically.
    """
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder:\n{folder}")

    unit_id = int(unit_id)
    pat = re.compile(r"unit[_\-]?0*(\d+)", re.IGNORECASE)

    candidates = []
    for p in folder.iterdir():
        if p.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        m = pat.search(p.stem)
        if not m:
            continue
        parsed = int(m.group(1))  # int("001") -> 1
        if parsed == unit_id:
            candidates.append(p)

    if not candidates:
        return None

    if prefer_contains:
        pref = [p for p in candidates if prefer_contains.lower() in p.name.lower()]
        if pref:
            candidates = pref

    return sorted(candidates, key=lambda p: len(p.name))[0]


def load_image(path: Path):
    return mpimg.imread(str(path))


def _matches_uid_in_name(path: Path, uid: int) -> bool:
    return re.search(rf"unit[_\-]?0*{int(uid)}\b", path.stem, flags=re.IGNORECASE) is not None


# =========================
# Load labels + features
# =========================

df_feat = pd.read_csv(features_csv)
df_lab = pd.read_csv(labels_csv)[["unit_id", "cell_type"]]

# Force unit_id type match (prevents silent merge weirdness)
df_feat["unit_id"] = pd.to_numeric(df_feat["unit_id"], errors="coerce").astype("Int64")
df_lab["unit_id"] = pd.to_numeric(df_lab["unit_id"], errors="coerce").astype("Int64")

df = df_feat.merge(df_lab, on="unit_id", how="inner")

# ALL units in labels (pyramidal + interneuron)
if MANUAL_UNIT_IDS:
    unit_ids = [int(u) for u in MANUAL_UNIT_IDS]
else:
    unit_ids = df["unit_id"].dropna().astype(int).sort_values().tolist()

print(f"Making sheets for ALL labeled units: {len(unit_ids)}")
metrics_cols = ["peak_to_trough_ms", "firing_rate_hz", "burst_index", "oscillation_score"]


# =========================
# Build one sheet per unit
# =========================

for uid in unit_ids:
    row = df.loc[df["unit_id"] == uid].iloc[0]

    # find existing images
    auto_wv_img_path = find_image_for_unit(auto_wv_dir, uid)
    rm_img_path = find_image_for_unit(ratemap_dir, uid, prefer_contains="rm")

    # load images (if found)
    auto_wv_img = load_image(auto_wv_img_path) if auto_wv_img_path else None
    rm_img = load_image(rm_img_path) if rm_img_path else None

    # sanity warnings (padding-safe)
    if auto_wv_img_path and not _matches_uid_in_name(auto_wv_img_path, uid):
        print(f"WARNING: auto/wv file mismatch? uid={uid} picked {auto_wv_img_path.name}")
    if rm_img_path and not _matches_uid_in_name(rm_img_path, uid):
        print(f"WARNING: ratemap file mismatch? uid={uid} picked {rm_img_path.name}")

    # layout: 2 rows x 2 cols
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # auto+wv panel  ✅ (this was missing in your broken script)
    if auto_wv_img is None:
        ax1.text(0.5, 0.5, f"No auto/wv image found for unit {uid}", ha="center", va="center")
        ax1.set_axis_off()
    else:
        ax1.imshow(auto_wv_img)
        ax1.set_title(f"Unit {uid}: waveform + ACG\n{auto_wv_img_path.name}")
        ax1.axis("off")

    # ratemap panel
    if rm_img is None:
        ax2.text(0.5, 0.5, f"No ratemap image found for unit {uid}", ha="center", va="center")
        ax2.set_axis_off()
    else:
        ax2.imshow(rm_img)
        ax2.set_title(f"Unit {uid}: ratemap\n{rm_img_path.name}")
        ax2.axis("off")

    # metrics text
    ax3.axis("off")
    lines = [
        f"unit_id: {uid}",
        f"cell_type (polygon): {row.get('cell_type', 'NA')}",
        f"phy_group: {row.get('phy_group', 'NA')}",
        f"meets_paper_isi: {row.get('meets_paper_isi', 'NA')}",
        "",
    ]
    for m in metrics_cols:
        val = row.get(m, "NA")
        lines.append(f"{m}: {val}")

    ax3.text(0.01, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=11)

    fig.suptitle(f"HCT inspection sheet — unit {uid}", y=0.98)
    fig.tight_layout()

    out_png = out_dir / f"unit_{uid}_inspection_sheet.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

print("\nDone. Sheets saved in:", out_dir)
