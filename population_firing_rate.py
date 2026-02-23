import numpy as np
from pathlib import Path
import spikeinterface.extractors as se
import pandas as pd
from typing import Literal
from scipy.stats import zscore
UnitTypes = Literal['pyramidal', 'good', 'all']


def calculate_population_firingrate(derivatives_base: Path, frame_rate: int = 25, sample_rate: int = 30000) -> None:
    """ The population firing rate script:

        Keeps pyramidal neurons only

        Pools all spikes

        Computes sliding 100 ms firing rate

        Normalizes it

        Uses it later to detect network bursts"""
    
    # Params
    window_length = 0.1 # in seconds: 100 ms
    sample_pad = (window_length/2) * sample_rate # 0.05 sec so I can use a 0.1 sec sliding window
    nSmallWindowsPerBig = 10
    window_interval = (window_length/nSmallWindowsPerBig) * sample_rate #10 ms between windows
        
    # Load data files
    kilosort_output_path = derivatives_base/ 'ephys'/ "concat_run"/"sorting"/ "sorter_output" 
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    # Here unit ids get filtered by unit_type
    unit_type = "pyramidal"
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)

    # Get unit ids with firing rate above 0.4 Hz
    unit_ids = get_firing_rate_units(derivatives_base, unit_ids)
    
    # Combine spikes for all units
    spike_train_allunits = []
    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)

        spike_train_allunits = np.append(spike_train_allunits, spike_train)
    
    
    # Edges for recording
    edges = np.arange(0, np.nanmax(spike_train_allunits) +window_interval, window_interval)
    spike_counts, _ = np.histogram(spike_train_allunits, bins=edges)
    spike_counts_avg= pd.Series(spike_counts).rolling(window=nSmallWindowsPerBig, center=True).mean().to_numpy()
    
    valid = ~np.isnan(spike_counts_avg)
    spike_counts_avg = spike_counts_avg[valid]

    # Convert to Hz per neuron
    n_units = len(unit_ids)
    spike_rate = (spike_counts_avg / n_units) / window_interval

    # Z-score
    spike_rate_z = zscore(spike_rate)
    
    return spike_rate_z
    
    

    

def load_unit_ids(derivatives_base: Path, unit_type: UnitTypes, unit_ids: list) -> list:
    """ Returns unit_ids, the unit_ids that we will create rmaps for"""
    if unit_type == 'good':
        good_units_path =derivatives_base/"ephys"/"concat_run"/"sorting"/"sorter_output"/"cluster_group.tsv"
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
    elif unit_type == 'pyramidal':
        pyramidal_units_path = derivatives_base/"analysis"/"cell_characteristics"/"unit_features"/"all_units_overview"/"pyramidal_units_2D.csv"
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
        print("Using pyramidal units")
    elif unit_type == "all":
        print("Using all units")
        unit_ids = unit_ids
    else:
        raise ValueError("unit_type not good, pyramidal, or all. Provide correct input")
    return unit_ids

def get_firing_rate_units(derivatives_base: Path, unit_ids: list):
    """ Gets the firing rate for each unit and removes units with firing rate below 0.4Hz"""
    metrics_df_path = derivatives_base/"analysis"/"cell_characteristics"/"unit_features"/"all_units_overview"/"unit_metrics.csv"
    metrics_df = pd.read_csv(metrics_df_path)
    
    metrics_df = metrics_df[metrics_df['unit_id'].isin(unit_ids)]
    
    unit_ids_filtered = []
    # Get firing rate for each unit
    for unit_id in unit_ids:
        firing_rate = metrics_df[metrics_df['unit_id'] == unit_id]['firing_rate'].values[0]
        if firing_rate >= 0.4:
            unit_ids_filtered.append(unit_id)
    return unit_ids_filtered
