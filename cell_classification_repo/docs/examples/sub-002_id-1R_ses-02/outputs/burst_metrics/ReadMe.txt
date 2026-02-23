burst_metrics_good_units.csv — README

This table contains burst-related metrics computed for Phy-curated good units only.
All metrics are derived from the autocorrelogram (ACG) using positive lags (0–50 ms).

Columns

unit_id
Unique cluster ID from Kilosort/Phy.

n_spikes
Total number of spikes used for the calculations.

n_nextISI_lt_2ms
Number of spikes whose next inter-spike interval is < 2 ms.

frac_nextISI_lt_2ms
Fraction of spikes with next ISI < 2 ms.
Used for the paper QC criterion (>1% → exclude).

paper_exclude_2ms
TRUE if frac_nextISI_lt_2ms > 0.01 (fails paper QC), else FALSE.

burst_index
Paper burst index:

ACG spike pairs in 0–10 ms
ACG spike pairs in 40–50 ms
ACG spike pairs in 40–50 ms
ACG spike pairs in 0–10 ms
	​


Measures relative burst probability vs baseline firing.

burst_index_norm01
Normalized version of burst_index:

𝑟
1
+
𝑟
1+r
r
	​


Squashes the ratio into [0, 1) for easier comparison.

acg_num_0_10
Number of ACG spike pairs in the burst window (0–10 ms).

acg_den_40_50
Number of ACG spike pairs in the baseline window (40–50 ms).

burst_contrast
Legacy / shape-based metric: high values indicate a sharp early ACG peak.
Sensitive to ACG shape, not absolute firing probability.