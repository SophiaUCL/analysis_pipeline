from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import spikeinterface as si


# -----------------------------
# Helpers
# -----------------------------

def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** int(np.ceil(np.log2(x)))

def _get_fs_hz_from_analyzer(sorting_analyzer) -> float:
    try:
        return float(sorting_analyzer.sampling_frequency)
    except Exception:
        pass
    try:
        return float(sorting_analyzer.recording.get_sampling_frequency())
    except Exception:
        pass
    return 30000.0

def _choose_w_bins(fc_hz: float, fmin_hz: float) -> int:
    """
    Paper sets w via criteria and requires W=2w power of 2. (Eq. 1)
    We'll implement the same spirit:
      - at least ~3 cycles of fmin per flank => w >= 3*fc/fmin
      - at least ~2 Hz resolution => W >= fc/2  => w >= fc/4
      - force W=2w to be power of 2
    Paper formula shown in Eq. 1. :contentReference[oaicite:1]{index=1}
    """
    w1 = int(np.ceil(3.0 * fc_hz / max(fmin_hz, 1e-9)))
    w2 = int(np.ceil(fc_hz / 4.0))
    w = max(w1, w2)
    W = _next_pow2(2 * w)
    return W // 2  # w such that W=2w is pow2

def _sigma_fast_bins(fc_hz: float, fmax_hz: float) -> float:
    """
    Paper Eq. 3 (in bins): sigma_fast = min(2, 134/(1.5*fmax) * fc/1000)
    With fc=1000 Hz this is min(2, 134/(1.5*fmax)).
    :contentReference[oaicite:2]{index=2}
    """
    return float(min(2.0, (134.0 / (1.5 * max(fmax_hz, 1e-9))) * (fc_hz / 1000.0)))

def _compute_ach_symmetric(
    spike_times_s: np.ndarray,
    bin_ms: float,
    w_bins: int,
    max_spikes: int = 200_000,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric ACH for lags [-w..w-1] with bin size bin_ms.

    Returns:
      lags_ms: centers for bins length 2*w_bins
      ach_counts: counts length 2*w_bins
    """
    st = np.asarray(spike_times_s, dtype=float)
    st = st[np.isfinite(st)]
    st.sort()

    # speed cap (important for very high-rate units)
    if st.size > max_spikes:
        rng = np.random.default_rng(rng_seed)
        st = rng.choice(st, size=max_spikes, replace=False)
        st.sort()

    w_ms = w_bins * bin_ms
    w_s = w_ms / 1000.0

    # bins for positive lags: (0..w_ms] in bin_ms steps
    edges_pos = np.arange(0.0, w_ms + bin_ms, bin_ms)  # ms edges, includes 0
    centers_pos = (edges_pos[:-1] + edges_pos[1:]) / 2  # ms centers

    counts_pos = np.zeros_like(centers_pos, dtype=float)

    if st.size >= 2:
        # forward scan: for each spike, look ahead within w_s
        diffs_ms: List[float] = []
        for i in range(st.size):
            j = i + 1
            while j < st.size:
                dt = st[j] - st[i]
                if dt > w_s:
                    break
                diffs_ms.append(dt * 1000.0)
                j += 1
        if len(diffs_ms) > 0:
            diffs_ms = np.asarray(diffs_ms, dtype=float)
            counts_pos, _ = np.histogram(diffs_ms, bins=edges_pos)

    # build symmetric ACH:
    # negative lags mirror positive (excluding 0 bin since we have only positive pairs)
    # We keep 2*w_bins bins: [-w..0) and [0..w)
    # For lag<0, counts match lag>0.
    counts_neg = counts_pos[::-1].copy()
    # centers for negative lags: -centers_pos reversed
    centers_neg = -centers_pos[::-1]

    lags_ms = np.concatenate([centers_neg, centers_pos])
    ach = np.concatenate([counts_neg, counts_pos]).astype(float)

    return lags_ms, ach


def _remove_central_peak_simple(
    lags_ms: np.ndarray,
    ach_smoothed: np.ndarray,
    baseline_window_ms: Tuple[float, float] = (200.0, 500.0),
    thresh_frac: float = 0.10,
) -> Tuple[np.ndarray, float]:
    """
    Paper uses a slow-smoothed ACH + curvature threshold to pick cut boundaries. :contentReference[oaicite:3]{index=3}
    For audit purposes we use a robust adaptive version:

    1) Estimate baseline from |lag| in [baseline_window_ms[0], baseline_window_ms[1]]
       (clipped to available lags).
    2) Starting from 0ms, expand left/right until ACH falls below:
         baseline + thresh_frac*(peak_at_0 - baseline)
    3) Overwrite that central region with the boundary value.

    Returns:
      ach_peakless
      cut_width_ms (half-width on one side)
    """
    ach = ach_smoothed.copy()

    abs_lag = np.abs(lags_ms)
    lo, hi = baseline_window_ms
    mask = (abs_lag >= lo) & (abs_lag <= hi)
    if not np.any(mask):
        # fallback: use farthest 20% of lags
        q = np.quantile(abs_lag, 0.8)
        mask = abs_lag >= q

    baseline = float(np.median(ach[mask]))

    # find index closest to 0
    i0 = int(np.argmin(np.abs(lags_ms)))
    peak0 = float(ach[i0])
    thr = baseline + thresh_frac * (peak0 - baseline)

    # expand to find boundaries where ach drops below thr
    left = i0
    right = i0
    while left > 0 and ach[left] > thr:
        left -= 1
    while right < len(ach) - 1 and ach[right] > thr:
        right += 1

    # overwrite [left..right] with boundary value (use mean of edges)
    fill_val = float(np.mean([ach[left], ach[right]]))
    ach[left:right + 1] = fill_val

    cut_width_ms = float(max(abs(lags_ms[left]), abs(lags_ms[right])))
    return ach, cut_width_ms


def oscillation_score_muresan2008(
    spike_times_s: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    bin_ms: float = 1.0,
    max_spikes: int = 200_000,
) -> Tuple[float, float, float, float, float, float]:
    """
    Returns:
      OS, f_osc, Mpeak, Mavg, w_ms, cut_width_ms

    Implements the key paper definition:
      OS = Mpeak / Mavg, where Mpeak is max magnitude within [fmin,fmax],
      and Mavg is mean magnitude across [0, fc/2].
    :contentReference[oaicite:4]{index=4}
    """
    fc_hz = 1000.0 / bin_ms  # correlogram frequency
    w_bins = _choose_w_bins(fc_hz=fc_hz, fmin_hz=fmin_hz)
    w_ms = w_bins * bin_ms

    lags_ms, ach = _compute_ach_symmetric(
        spike_times_s, bin_ms=bin_ms, w_bins=w_bins, max_spikes=max_spikes
    )

    # Step 2: fast Gaussian smoothing (Eq. 3)
    sigma_fast = _sigma_fast_bins(fc_hz=fc_hz, fmax_hz=fmax_hz)
    ach_smooth = gaussian_filter1d(ach, sigma=sigma_fast, mode="nearest")

    # Step 3: remove central peak (audit version; adaptive)
    ach_peakless, cut_width_ms = _remove_central_peak_simple(lags_ms, ach_smooth)

    # Step 4: apply Blackman window then FFT magnitude
    window = np.blackman(len(ach_peakless))
    x = ach_peakless * window

    # FFT -> only nonnegative freqs
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(x), d=(bin_ms / 1000.0))  # Hz

    # Step 5: Mavg over [0..fc/2]
    Mavg = float(np.mean(mag))

    # Mpeak in band
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    if not np.any(band) or Mavg <= 0 or not np.isfinite(Mavg):
        return np.nan, np.nan, np.nan, Mavg, w_ms, cut_width_ms

    idx_peak = int(np.argmax(mag[band]))
    band_freqs = freqs[band]
    band_mag = mag[band]
    f_osc = float(band_freqs[idx_peak])
    Mpeak = float(band_mag[idx_peak])

    OS = float(Mpeak / Mavg) if Mavg > 0 else np.nan
    return OS, f_osc, Mpeak, Mavg, w_ms, cut_width_ms


def plot_oscillation_diagnostic(
    unit_id: int,
    spike_times_s: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    out_path: Path,
    bin_ms: float = 1.0,
    max_spikes: int = 200_000,
) -> None:
    # recompute internals for plotting (kept simple/explicit)
    fc_hz = 1000.0 / bin_ms
    w_bins = _choose_w_bins(fc_hz=fc_hz, fmin_hz=fmin_hz)

    lags_ms, ach = _compute_ach_symmetric(
        spike_times_s, bin_ms=bin_ms, w_bins=w_bins, max_spikes=max_spikes
    )

    sigma_fast = _sigma_fast_bins(fc_hz=fc_hz, fmax_hz=fmax_hz)
    ach_smooth = gaussian_filter1d(ach, sigma=sigma_fast, mode="nearest")
    ach_peakless, cut_width_ms = _remove_central_peak_simple(lags_ms, ach_smooth)

    window = np.blackman(len(ach_peakless))
    spec = np.fft.rfft(ach_peakless * window)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(ach_peakless), d=(bin_ms / 1000.0))

    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    Mavg = np.mean(mag) if len(mag) else np.nan
    if np.any(band):
        f_osc = freqs[band][np.argmax(mag[band])]
    else:
        f_osc = np.nan

    fig, axs = plt.subplots(2, 1, figsize=(9, 7))

    # ACH view (smoothed + peakless)
    axs[0].plot(lags_ms, ach_smooth, label="ACH smoothed")
    axs[0].plot(lags_ms, ach_peakless, label="ACH peakless")
    axs[0].axvspan(-cut_width_ms, cut_width_ms, alpha=0.15, label="central cut")
    axs[0].set_title(f"Unit {unit_id} ACH (bin={bin_ms} ms, sigma_fast={sigma_fast:.2f} bins)")
    axs[0].set_xlabel("Lag (ms)")
    axs[0].set_ylabel("Count (arb.)")
    axs[0].legend(loc="upper right")

    # Spectrum view
    axs[1].plot(freqs, mag, label="FFT magnitude")
    axs[1].axvspan(fmin_hz, fmax_hz, alpha=0.15, label=f"band {fmin_hz}-{fmax_hz} Hz")
    axs[1].axvline(f_osc, linestyle="--", label=f"f_osc={f_osc:.2f} Hz")
    axs[1].axhline(Mavg, linestyle=":", label="Mavg")
    axs[1].set_xlim(0, min(150, freqs.max() if len(freqs) else 150))
    axs[1].set_title("Spectrum of peakless ACH")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def audit_oscillation_score(
    derivatives_base: str,
    project_base: str,
    fmin_hz: float = 5.0,
    fmax_hz: float = 12.0,
    bin_ms: float = 1.0,
    max_spikes: int = 200_000,
    make_plots: bool = True,
    n_plot_units_each: int = 5,
) -> Path:
    """
    Reads spike trains from sorting_analyzer (no changes to derivatives).
    Writes CSV + diagnostic plots to project_base/audits/oscillation_score.
    """
    unit_features_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features")
    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")
    if not os.path.exists(analyzer_path):
        raise FileNotFoundError(f"Missing sorting_analyzer:\n{analyzer_path}")

    print("Loading sorting_analyzer (read-only):")
    print(" ", analyzer_path)
    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_path)
    sorting = sorting_analyzer.sorting
    fs_hz = _get_fs_hz_from_analyzer(sorting_analyzer)
    print("Sampling frequency (Hz):", fs_hz)

    out_dir = Path(project_base) / "audits" / "oscillation_score"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "oscillation_score_audit.csv"

    rows = []
    for uid in sorting.unit_ids:
        st_samples = sorting.get_unit_spike_train(unit_id=uid)
        st_s = np.asarray(st_samples, dtype=float) / fs_hz

        OS, f_osc, Mpeak, Mavg, w_ms, cut_ms = oscillation_score_muresan2008(
            st_s,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            bin_ms=bin_ms,
            max_spikes=max_spikes,
        )

        rows.append(
            dict(
                unit_id=int(uid),
                n_spikes=int(len(st_s)),
                oscillation_score=OS,
                f_osc_hz=f_osc,
                Mpeak=Mpeak,
                Mavg=Mavg,
                band_fmin_hz=float(fmin_hz),
                band_fmax_hz=float(fmax_hz),
                bin_ms=float(bin_ms),
                ach_window_w_ms=float(w_ms),
                central_cut_halfwidth_ms=float(cut_ms),
                max_spikes_used=int(min(len(st_s), max_spikes)),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("\nSaved audit CSV to:")
    print(" ", out_csv)

    finite = df[np.isfinite(df["oscillation_score"])]
    print("\nOscillation score summary (finite only):")
    if len(finite) > 0:
        print(finite["oscillation_score"].describe())

    if make_plots and len(finite) > 0:
        top = finite.sort_values("oscillation_score", ascending=False).head(n_plot_units_each)["unit_id"].tolist()
        bot = finite.sort_values("oscillation_score", ascending=True).head(n_plot_units_each)["unit_id"].tolist()
        plot_units = top + bot

        for uid in plot_units:
            st_samples = sorting.get_unit_spike_train(unit_id=int(uid))
            st_s = np.asarray(st_samples, dtype=float) / fs_hz
            plot_oscillation_diagnostic(
                unit_id=int(uid),
                spike_times_s=st_s,
                fmin_hz=fmin_hz,
                fmax_hz=fmax_hz,
                out_path=fig_dir / f"unit_{int(uid):04d}_oscscore_diagnostic.png",
                bin_ms=bin_ms,
                max_spikes=max_spikes,
            )

        print(f"\nSaved {len(plot_units)} diagnostic plots to:")
        print(" ", fig_dir)

    return out_csv


if __name__ == "__main__":

    # READ ONLY
    derivatives_base = (
        r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\Data\Honeycomb_Maze_Task"
        r"\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    )

    # WRITE OUTPUTS HERE
    project_base = (
        r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Chiara\cell_classification_projects\HCT"
        r"\sub-002_id-1R\ses-02_date-11092025"
    )

    # Theta band by default (edit if needed)
    audit_oscillation_score(
        derivatives_base=derivatives_base,
        project_base=project_base,
        fmin_hz=5.0,
        fmax_hz=12.0,
        bin_ms=1.0,
        max_spikes=200_000,
        make_plots=True,
        n_plot_units_each=5,
    )
