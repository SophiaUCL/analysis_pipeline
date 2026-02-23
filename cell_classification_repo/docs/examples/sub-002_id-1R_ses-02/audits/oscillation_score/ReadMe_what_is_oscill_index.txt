### Oscillation score (spike-time rhythmicity)

The oscillation score quantifies how strongly a neuron’s spike timing is rhythmically
modulated at a given frequency, independently of firing rate and bursting.
It measures oscillatory structure in the spike train rather than voltage oscillations
(e.g. LFP).

Our implementation is based on the method introduced by:

Mureșan, R. C., Jurjuţ, O. F., Moca, V. V., Singer, W., & Nikolić, D.  
*The oscillation score: an efficient method for estimating oscillation strength in neuronal activity.*  
Journal of Neurophysiology, 99, 1333–1353 (2008).

The oscillation score is computed from the autocorrelation structure of each unit’s
spike train and reflects the strength of rhythmic firing at a chosen frequency band.

---

### Conceptual overview

If a neuron fires rhythmically at frequency \( f \), its spike train exhibits repeated
inter-spike intervals at multiples of the oscillation period \( T = 1/f \).
This periodic structure appears as regularly spaced peaks in the autocorrelation
histogram (ACH) of the spike train.

The oscillation score measures the strength of this periodic structure by:
1. Computing the autocorrelation histogram of the spike train
2. Removing non-oscillatory components (rate, refractory effects)
3. Analyzing the frequency content of the remaining oscillatory structure

---

### Step-by-step computation

For each unit, the oscillation score is computed as follows:

#### 1. Compute the autocorrelation histogram (ACH)

- A **symmetric autocorrelation histogram** is computed from the spike train
- Time lags span a window \([-W, +W]\) ms, where \( W \) is chosen following the
  criteria described in Mureșan et al. (2008)
- The ACH is computed using **1 ms bins**
- For computational efficiency, very large spike trains may be subsampled

The ACH contains contributions from:
- firing rate
- refractory period
- burst firing
- oscillatory structure

---

#### 2. Smooth the ACH (fast Gaussian smoothing)

- The ACH is smoothed using a **fast Gaussian kernel**
- The kernel width is chosen following Eq. (3) in Mureșan et al. (2008) and depends on
  the upper frequency bound of the oscillation band
- This step suppresses sampling noise while preserving oscillatory structure

---

#### 3. Remove the central peak

- The large peak around zero lag reflects firing rate, refractory effects, and burst
  structure rather than oscillations
- Including this peak would dominate the frequency spectrum and bias the oscillation
  estimate

To remove it:
- A baseline ACH level is estimated from large time lags
- The region around zero lag where the ACH exceeds a fraction of the peak amplitude
  is identified
- This central region is replaced with baseline values, yielding a **peakless ACH**

This step ensures that the oscillation score is independent of firing rate and bursting.

---

#### 4. Compute the frequency spectrum

- A Blackman window is applied to the peakless ACH
- The **FFT magnitude spectrum** of the ACH is computed
- Frequencies correspond to rhythmic structure in spike timing

---

#### 5. Identify oscillatory power in the frequency band

- A frequency band \([f_{\min}, f_{\max}]\) is selected (e.g. theta: 5–12 Hz)
- Within this band, the **maximum spectral magnitude** is identified:
  \[
  M_{\text{peak}}
  \]

---

#### 6. Normalize by average spectral magnitude

- The mean magnitude of the FFT spectrum across all frequencies is computed:
  \[
  M_{\text{avg}}
  \]

This represents the background spectral level and provides normalization.

---

#### 7. Oscillation score

The oscillation score is defined as:
\[
\text{Oscillation Score} =
\frac{M_{\text{peak}}}{M_{\text{avg}}}
\]

This yields a dimensionless measure of oscillatory strength.

---

### Interpretation

- Oscillation score ≈ 1  
  → no detectable rhythmic modulation beyond background

- Oscillation score > 1  
  → spike timing is rhythmically modulated in the chosen frequency band

- Larger values indicate stronger and more consistent oscillatory firing

Importantly, oscillation score:
- is independent of mean firing rate
- is distinct from burst firing
- captures rhythmic structure specific to individual neurons

---

### Outputs

The oscillation score audit produces:

- **oscillation_score**  
  Oscillation score as defined above

- **f_osc_hz**  
  Frequency at which the peak spectral magnitude occurs within the chosen band

- **Mpeak**  
  Peak spectral magnitude within the band

- **Mavg**  
  Mean spectral magnitude across frequencies

- **ach_window_w_ms**  
  Half-width of the ACH window used (ms)

- **central_cut_halfwidth_ms**  
  Half-width of the removed central ACH peak (ms)

These values are saved to:
