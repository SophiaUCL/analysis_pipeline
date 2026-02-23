from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import json

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface as si


# ========= EDIT ONLY THESE =========
base_path = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_1g_Task\rawdata")
out_base = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\lfp_projects\Honeycomb_1g_Task")

subject = "sub-001_id-2H"
session = "ses-02_date-12022026"
# ===================================

session_dir = base_path / subject / session
ephys_dir = session_dir / "ephys"

out_dir = out_base / subject / session / "lfp_concat"
out_dir.mkdir(parents=True, exist_ok=True)
print("Writing output to:", out_dir)

out_file = out_dir /  "lfp_CARmed_0p5_300Hz_1kHz.bin"
meta_file = out_dir / "lfp_CARmed_0p5-300Hz_1kHz_meta.json"


# 1) Find run folders that look like "..._g0", "..._g1", ...
run_dirs = sorted([p for p in ephys_dir.iterdir() if p.is_dir() and "_g" in p.name])
print("Runs found:", [p.name for p in run_dirs])

processed = []

for run in run_dirs:
    imec0_dirs = [p for p in run.iterdir() if p.is_dir() and p.name.endswith("_imec0")]
    if not imec0_dirs:
        print("Skipping (no imec0):", run.name)
        continue
    imec0 = imec0_dirs[0]

    # Load AP stream (important: stream_id must match SpikeInterface's reported id)
    rec = se.read_spikeglx(imec0, stream_id="imec0.ap")

    # CAR median (global median reference)
    rec = spre.common_reference(rec, reference="global", operator="median")

    # Bandpass for LFP
    rec = spre.bandpass_filter(rec, freq_min=0.5, freq_max=300.0)

    # Resample to 1000 Hz
    rec = spre.resample(rec, 1000)

    processed.append(rec)
    print("Processed:", run.name, "| duration(s)=", rec.get_total_duration())


assert len(processed) > 0, "No runs were processed!"

rec_concat = si.concatenate_recordings(processed)

print("Concatenated recording:")
print("  Duration (s):", rec_concat.get_total_duration())
print("  Sampling rate (Hz):", rec_concat.get_sampling_frequency())
print("  Number of channels:", rec_concat.get_num_channels())

si.write_binary_recording(
    rec_concat,
    file_paths=out_file,
    dtype="int16",
    chunk_duration="1s"
)
print("Saved LFP to:", out_file)

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




