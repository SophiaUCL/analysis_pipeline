import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.fft import rfft, rfftfreq


def oscillation_score(
    spike_times_s: np.ndarray,
    bin_ms: float = 2.0,
    max_lag_ms: float = 1000.0,
    sigma_ms: float = 4.0,
    remove_ms: float = 10.0,
    fmin_hz: float = 4.0,
    fmax_hz: float = 12.0,
    min_spikes: int = 200,
):
    """
    Compute oscillation score following Mureşan et al. (2008).

    Steps:
      1) Build symmetric ACG histogram (lags in [-max_lag_ms, +max_lag_ms]) with bin_ms bins
      2) Smooth ACG with a Gaussian (sigma_ms)
      3) Remove central peak (|lag| <= remove_ms)
      4) Take magnitude spectrum of ACG via FFT
      5) Oscillation score = peak magnitude in [fmin_hz, fmax_hz] / mean magnitude over all freqs (excluding DC)

    Returns
    -------
    osc_score : float
    peak_freq_hz : float
    M_peak : float
    M_avg : float
    """
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    spike_times_s = spike_times_s[np.isfinite(spike_times_s)]
    spike_times_s.sort()

    if spike_times_s.size < min_spikes:
        return np.nan, np.nan, np.nan, np.nan

    # ---------- ACG (symmetric) ----------
    spike_times_ms = spike_times_s * 1000.0
    diffs = []

    n = spike_times_ms.size
    for i in range(n):
        j = i + 1
        while j < n:
            d = spike_times_ms[j] - spike_times_ms[i]
            if d > max_lag_ms:
                break
            diffs.append(d)
            diffs.append(-d)
            j += 1

    diffs = np.asarray(diffs, dtype=float)

    edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
    acg, _ = np.histogram(diffs, bins=edges)
    lags_ms = edges[:-1] + bin_ms / 2.0

    # ---------- Smooth ----------
    sigma_bins = sigma_ms / bin_ms
    acg_smooth = gaussian_filter1d(acg.astype(float), sigma=sigma_bins)

    # ---------- Remove central peak ----------
    central_mask = np.abs(lags_ms) <= remove_ms
    acg_smooth[central_mask] = 0.0

    # ---------- Spectrum ----------
    dt_s = bin_ms / 1000.0
    spec = np.abs(rfft(acg_smooth))
    freqs = rfftfreq(acg_smooth.size, d=dt_s)

    # exclude DC (0 Hz)
    valid = freqs > 0
    freqs = freqs[valid]
    spec = spec[valid]

    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    if not np.any(band):
        return np.nan, np.nan, np.nan, np.nan

    M_peak = float(np.max(spec[band]))
    peak_freq = float(freqs[band][np.argmax(spec[band])])
    M_avg = float(np.mean(spec)) if spec.size > 0 else np.nan

    osc_score = float(M_peak / M_avg) if np.isfinite(M_avg) and M_avg > 0 else np.nan
    return osc_score, peak_freq, M_peak, M_avg
