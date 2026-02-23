from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, decimate

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

base_path = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_1g_Task\rawdata")
subject = "sub-001_id-2H"
session = "ses-02_date-12022026"

session_dir = base_path / subject / session
print("Session dir:", session_dir)
print("Contents:", [p.name for p in session_dir.iterdir()])

ephys_dir = session_dir / "ephys"
print("Ephys contents:", [p.name for p in ephys_dir.iterdir()])

run = ephys_dir / "ses_01_g00"   # <-- change to whatever exists
print([p.name for p in run.iterdir()])

imec0 = [p for p in run.iterdir() if p.is_dir() and p.name.endswith("_imec0")][0]
ap_bin = list(imec0.glob("*.ap.bin"))[0]
ap_meta = list(imec0.glob("*.ap.meta"))[0]
print("AP bin:", ap_bin)
print("AP meta:", ap_meta)

ap_meta = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_1g_Task\rawdata\sub-002_id-1R\ses-02_date-11092025\ephys\ses_01_g00\ses_01_g0_imec0\ses_01_g0_t0.imec0.ap.meta")

meta = {}

with open(ap_meta, "r") as f:
    for line in f:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            meta[key]=value
print("Sampling rate (Hz):", meta["imSampRate"])
print("Number of saved channels:", meta["nSavedChans"])

# memory-map the binary file (no loading into RAM)
mm = np.memmap(
    r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task\rawdata\sub-002_id-1R\ses-02_date-11092025\ephys\ses_01_g00\ses_01_g0_imec0\ses_01_g0_t0.imec0.ap.bin",
    dtype=np.int16,
    mode="r"
)

# total number of samples in the file
n_samples_total = mm.size // 385

print("Total samples:", n_samples_total)
print("Total duration (s):", n_samples_total / 30000)

# reshape into (samples, channels) without copying
mm2 = mm.reshape(n_samples_total, 385)

# 1 second chunk = 30,000 samples at 30 kHz
fs = 30000
n_1s = fs * 1

chunk = mm2[0:n_1s, :]   # first second, all channels

print("Chunk shape:", chunk.shape)
print("Chunk dtype:", chunk.dtype)
print("Example values (first 5 samples of channel 0):", chunk[0:5, 0])

print("Keys in meta include imAiRangeMax?", "imAiRangeMax" in meta)
print("imAiRangeMax:", meta.get("imAiRangeMax"))
print("Keys include imroTbl?", "imroTbl" in meta)

# print only a SHORT preview so we don't flood the terminal
imro_preview = meta.get("imroTbl", "")[:200]
print("imroTbl preview:", imro_preview)

# convert raw int16 to microvolts (NP2.0)
uV_per_bit = (0.62 * 1e6) / 32768
chunk_uV = chunk.astype(np.float32) * uV_per_bit

print("uV per bit:", uV_per_bit)
print("Example values in µV:", chunk_uV[0:5, 0])

fs = 30000
# choose channels to test (ignore last channel 384 for now; it's often sync)
test_chans = [0, 50, 100, 150, 200]

# take 10 seconds so filtering is stable (not just 1 second)
n_sec = 10
n_samp = int(fs * n_sec)
chunk10 = mm2[0:n_samp,test_chans].astype(np.float32) * uV_per_bit

# --- Global median CAR ---
# median across channels at each timepoint (shape: (time,))
global_median = np.median(chunk10, axis=1)

# subtract from every channel (broadcasts automatically)
chunk10_car = chunk10 - global_median[:, None]

print("Shapes:", chunk10.shape, global_median.shape, chunk10_car.shape)

def bandpass_lfp(x_uV, fs, low=0.5, high=300.0, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x_uV, axis=0)

lfp10 = bandpass_lfp(chunk10_car, fs=fs, low=0.5, high=300.0, order=4)

# downsample from 30 kHz → 1 kHz
lfp10_ds = decimate(
    lfp10,
    q=30,               # downsampling factor
    axis=0,
    ftype="iir",
    zero_phase=True
)

fs_lfp = fs / 30
print("New LFP sampling rate:", fs_lfp)
print("Downsampled shape:", lfp10_ds.shape)


