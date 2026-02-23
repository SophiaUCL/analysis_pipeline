from pathlib import Path
import json

# ===== EDIT THESE =====
out_base = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\lfp_projects\HCT")
subject = "sub-002_id-1R"
session = "ses-02_date-11092025"
# ======================

out_dir = out_base / subject / session / "lfp_concat"
out_file = out_dir / "lfp_CARmed_0p5_300Hz_1kHz.bin"
meta_file = out_dir / "lfp_CARmed_0p5_300Hz_1kHz_meta.json"

assert out_file.exists(), f"Missing LFP bin: {out_file}"

meta = {
    "sampling_frequency": 1000,
    "num_channels": 384,
    "dtype": "int16",
    "reference": "global_median_CAR",
    "bandpass_hz": [0.5, 300],
    "downsampled_from_hz": 30000,
    "units": "microvolts",
    "description": "Concatenated LFP across all trials (AP-derived)"
}

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=4)

print("Saved metadata to:", meta_file)
