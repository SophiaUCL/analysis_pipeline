This table audits refractory-period violations for Phy-curated good units, following the criterion described in the paper

 Cells with greater than 1% of spikes within the first 2 ms of the spike autocorrelogram were excluded from further analysis.
 

Rather than using a full autocorrelogram, this audit implements an equivalent and standard operationalization

the fraction of spikes whose next inter-spike interval (ISI) is  2 ms.

### Column descriptions

- `unit_id`
    
    Cluster  unit identifier (matches Kilosort  Phy IDs).
    
- `n_spikes`
    
    Total number of spikes assigned to the unit across the session.
    
- `n_nextISI_lt_threshold`
    
    Number of spikes whose next inter-spike interval is shorter than the threshold (2 ms).
    
    These are refractory-period violations.
    
- `frac_nextISI_lt_threshold`n_spikesn_nextISI_lt_threshold
    
    Fraction of spikes violating the refractory criterion
    
    n_nextISI_lt_thresholdn_spikesfrac{text{n_nextISI_lt_threshold}}{text{n_spikes}}
    
- `threshold_ms`
    
    ISI threshold used for the audit (2 ms).
    
- `paper_exclude`
    
    Boolean flag indicating whether the unit would be excluded under the paper’s criterion
    
    `TRUE` if `frac_nextISI_lt_threshold  0.01` (i.e.  1%).
    
- `exclude_frac`
    
    Fractional cutoff used for exclusion (0.01 = 1%).