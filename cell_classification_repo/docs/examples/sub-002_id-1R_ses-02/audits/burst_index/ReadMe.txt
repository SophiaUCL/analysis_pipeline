### Burst index (autocorrelation-based)

The burst index quantifies the tendency of a unit to fire spikes in rapid succession
("bursts") and is computed from the autocorrelogram (ACG) of each unit’s spike train.

All burst index metrics are computed from positive-lag autocorrelograms
(0–50 ms) using 1 ms bins.

#### Columns

- **unit_id**  
  Unique unit identifier from Kilosort / SpikeInterface. This key is used to merge
  waveform, quality, and firing metrics across analyses.

- **n_spikes**  
  Total number of spikes used to compute the autocorrelogram. Units with very low
  spike counts may yield unstable burst index estimates and can be excluded or
  flagged in downstream analyses.

- **acg_pairs_0_10**  
  Number of spike pairs with inter-spike intervals between 0 and 10 ms. This window
  captures short-latency spike clustering and is sensitive to burst firing.

- **acg_pairs_40_50**  
  Number of spike pairs with inter-spike intervals between 40 and 50 ms. This window
  is used as a baseline reference, far from refractory effects, approximating
  chance-level spike pairing given the unit’s firing rate.

- **burst_index_paper**  
  Burst index computed following the definition used in the literature:

  \[
  \text{burst index} =
  \frac{\#\text{ spike pairs in }0\text{–}10\text{ ms}}
       {\#\text{ spike pairs in }40\text{–}50\text{ ms}}
  \]

  Values near 1 indicate no excess bursting beyond baseline firing.
  Values greater than 1 indicate bursty firing.
  Values below 1 indicate regular or anti-bursty firing.

- **burst_index_legacy_normdiff**  
  A legacy burst metric retained for comparison with older code. This metric is
  based on a normalized difference between short-lag and baseline ACG counts and
  does not correspond to the paper definition -> I can modify burst index code so it matches John's paper ? 
