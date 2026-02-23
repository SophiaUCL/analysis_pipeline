from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface as si

# ====== EDIT THESE ======
out_base = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\lfp_projects\HCT")
subject = "sub-002_id-1R"
session = "ses-02_date-11092025"

# Kilosort sorter_output folder (for geometry)
ks_sorter_output = Path(r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials\ephys\concat_run\sorting\sorter_output")
# =======================

# Where the concatenated LFP file lives
out_dir = out_base / subject / session / "lfp_concat"
lfp_bin = out_dir / "lfp_CARmed_0p5_300Hz_1kHz.bin"

# Where we save plots from THIS script
plots_dir = out_dir / "lfp_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

print("LFP file:", lfp_bin)
print("Plots dir:", plots_dir)
print("Kilosort sorter_output:", ks_sorter_output)

# LFP file format assumptions
FS = 1000
NCH = 384
DTYPE = "int16"

# Load as SpikeInterface recording handle (lazy reading)
rec = si.read_binary(
    file_paths=lfp_bin,
    sampling_frequency=FS,
    num_channels=NCH,
    dtype=DTYPE
)

print("Loaded LFP. Duration (s):", rec.get_total_duration())
print("Fs:", rec.get_sampling_frequency(), "| Channels:", rec.get_num_channels())

# Quick check: confirm geometry files exist
pos_path = ks_sorter_output / "channel_positions.npy"
shank_path = ks_sorter_output / "channel_shanks.npy"
print("Geometry files exist?",
      pos_path.exists(), shank_path.exists())
print("pos_path:", pos_path)
print("shank_path:", shank_path)

# ===== Step 2: Load geometry =====
pos = np.load(pos_path)        # shape (384, 2): x,y in micrometers
shank = np.load(shank_path)    # shape (384,): shank id (usually 0..3)

print("pos shape:", pos.shape)
print("shank shape:", shank.shape)
print("Unique shanks:", np.unique(shank))

# Plot geometry (x,y)
plt.figure(figsize=(5, 7))
plt.scatter(pos[:, 0], pos[:, 1], s=10)
plt.gca().invert_yaxis()  # makes "deeper" go downward visually (common convention)
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Channel geometry (all channels)")
plt.tight_layout()
plt.savefig(plots_dir / "geometry_all_channels.png", dpi=300)
plt.show()
plt.close()

# ===== Step 3A: Theta power per channel =====
def band_power(rec, fs, seconds=60, band=(6, 10), start_s=60, max_ch=384):
    """
    Returns band power per channel (array length = max_ch).
    Uses FFT on a chunk of length `seconds` starting at `start_s`.
    """
    n = int(fs * seconds)
    start = int(fs * start_s)
    chans = np.arange(max_ch)

    x = rec.get_traces(start_frame=start, end_frame=start + n, channel_ids=chans).astype(np.float32)
    x -= x.mean(axis=0, keepdims=True)

    freqs = np.fft.rfftfreq(n, d=1/fs)
    X = np.fft.rfft(x, axis=0)
    P = (np.abs(X) ** 2) / n

    f0, f1 = band
    mask = (freqs >= f0) & (freqs <= f1)
    bp = P[mask, :].sum(axis=0)
    return bp


theta_pow = band_power(rec, fs=FS, seconds=60, band=(6, 10), start_s=60, max_ch=NCH)
ranked = np.argsort(theta_pow)[::-1]

top20 = ranked[:20].tolist()
best_ch = int(ranked[0])
best_shank = int(shank[best_ch])
best_xy = pos[best_ch]

print("Top 20 theta channels:", top20)
print("Best theta channel:", best_ch, "| shank:", best_shank, "| x,y:", best_xy)

plt.figure(figsize=(10, 4))
plt.plot(theta_pow)
plt.xlabel("Channel index")
plt.ylabel("Theta power (6–10 Hz, a.u.)")
plt.title("Theta power across channels")
plt.tight_layout()
plt.savefig(plots_dir / "theta_power_across_channels.png", dpi=300)
plt.show()
plt.close()

topN = 20
top_ch = ranked[:topN]

plt.figure(figsize=(5, 7))
plt.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.25)
plt.scatter(pos[top_ch, 0], pos[top_ch, 1], s=40)  # highlighted
plt.gca().invert_yaxis()
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title(f"Top {topN} theta channels over probe geometry")
plt.tight_layout()
plt.savefig(plots_dir / "theta_top20_on_geometry.png", dpi=300)
plt.show()
plt.close()

# ===== Step 4: Select local channels around best theta channel =====
depth_window_um = 200  # you can change to 150 or 300 if needed

same_shank = np.where(shank == best_shank)[0]
y = pos[:, 1]
best_y = y[best_ch]

local_idx = same_shank[np.abs(y[same_shank] - best_y) <= depth_window_um]

# Rank ONLY those local channels by theta power
local_sorted = sorted(local_idx.tolist(), key=lambda ch: theta_pow[ch], reverse=True)

chans8 = local_sorted[:8]
chans5 = local_sorted[:5]

print(f"Local channels on shank {best_shank} within ±{depth_window_um}µm of y={best_y}: {len(local_sorted)}")
print("Selected 8 local channels:", chans8)
print("Selected 5 local channels:", chans5)

plt.figure(figsize=(5, 7))

# faint background: only same shank
plt.scatter(pos[same_shank, 0], pos[same_shank, 1], s=10, alpha=0.25, label=f"shank {best_shank}")

# highlight selected channels
plt.scatter(pos[chans8, 0], pos[chans8, 1], s=60, label="selected (top theta, local)")

plt.gca().invert_yaxis()
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title(f"Selected local theta channels (shank {best_shank})")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(plots_dir / "selected_local_theta_channels_geometry.png", dpi=300)
plt.show()
plt.close()

def plot_1s_from_rec(rec, fs, chans, start_s=60, title="1s snippet", save_path=None, offset_uV=200):
    start = int(fs * start_s)
    traces = rec.get_traces(
        start_frame=start,
        end_frame=start + fs,
        channel_ids=chans
    ).astype(np.float32)

    t_ms = np.arange(traces.shape[0])  # fs=1000 → ms

    plt.figure(figsize=(10, 5))
    off = 0.0

    for i, ch in enumerate(chans):
        plt.plot(t_ms, traces[:, i] + off, lw=1)
        # label each trace on the right
        plt.text(
            t_ms[-1] + 5,
            traces[-1, i] + off,
            f"ch {ch}",
            fontsize=8,
            verticalalignment="center"
        )
        off += offset_uV

    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (µV) + offset")
    plt.title(f"{title} (start {start_s}s)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print("Saved:", save_path)

    plt.show()
    plt.close()


plot_1s_from_rec(
    rec, fs=FS, chans=chans8, start_s=60,
    title="LFP: selected local theta channels",
    save_path=plots_dir / "lfp_selected_local_theta_1s.png"
)

plot_1s_from_rec(
    rec, fs=FS, chans=chans5, start_s=60,
    title="LFP: selected local theta channels (top 5)",
    save_path=plots_dir / "lfp_selected_local_theta_top5_1s.png"
)
