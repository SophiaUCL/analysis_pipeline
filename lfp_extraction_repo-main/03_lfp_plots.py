import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import spikeinterface as si

# --- point to your output file ---
out_base = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\lfp_projects\Honeycomb_1g_Task")
subject = "sub-001_id-2H"
session = "ses-02_date-12022026"

out_dir = out_base / subject / session / "lfp_concat"
out_file = out_dir / "lfp_CARmed_0p5_300Hz_1kHz.bin"
plots_dir = out_dir / "lfp_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

print("QC plots will be saved to:", plots_dir)

fs = 1000
num_channels = 384
dtype = "int16"

rec = si.read_binary(
    file_paths=out_file,
    sampling_frequency=fs,
    num_channels=num_channels,
    dtype=dtype
)

print("Loaded:", out_base.name)
print("Duration (s):", rec.get_total_duration())
print("Fs:", rec.get_sampling_frequency())
print("Channels:", rec.get_num_channels())

def plot_1s_multich(traces, fs, chans, title="1s LFP snippet", save_path=None):
    """
    traces: np.array (samples x channels)
    fs: sampling rate (Hz)
    chans: list of channel indices
    """
    t = np.arange(traces.shape[0]) / fs * 1000  # ms

    plt.figure(figsize=(10, 5))
    offset = 0.0
    for i, ch in enumerate(chans):
        plt.plot(t, traces[:, i] + offset)
        offset += 200  # µV offset for readability

    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV) + offset")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print("Saved plot to:", save_path)
    plt.show()

# plot 1 second from the start
chans = [0, 50, 100, 150, 200, 250, 300, 350]
snippet = rec.get_traces(start_frame=0, end_frame=fs, channel_ids=chans)  # fs=1000 => 1s

plot_1s_multich(snippet, fs, chans, title="LFP sanity check (1s)", save_path= plots_dir / "lfp_sanity_check_1s.png")



