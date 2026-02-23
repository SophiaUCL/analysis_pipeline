import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
from pathlib import Path
from unit_features.utils import load_unit_ids, get_spike_train_s, make_plot, get_goal_1_end_times, get_total_trial_length, get_trials_length_df

UnitTypes = Literal['pyramidal', 'good', 'all']
Tasks = Literal['hct', 'spatiotemporal']

def plot_spikecount_over_trials(derivatives_base: Path, 
                                unit_type: UnitTypes, 
                                trials_to_include: list, 
                                task: Tasks,
                                last_trial_openfield: bool = False, 
                                sample_rate = 30000):
    r"""For each unit, creates a plot of its spikecount throughout time.
    Also indicates the trials

    Args:
        derivatives_base (Path): Path to derivatives folder
        unit_type (all, pyramidal, or good): Which unit type to use
        trials_to_include (list): trials to include
        task (hct or spatiotemp): which task we're running
        sample_rate (int, 30000): sample rate of recording


    Raises:
    ValueError
        - If `unit_type` is not one of {'good', 'pyramidal', 'all'} in `load_unit_ids`.
        - If the number of rows in `alltrials_trialday.csv` does not match the number 
        of trials in `trials_to_include` in `get_goal_1_end_times`.

    FileNotFoundError
        - If `trials_length.csv` is missing in `get_trials_length_df`.
    
    Outputs:
        derivatives_base\analysis\cell_characteristics\unit_features\spikecount_over_trials\unit_{}_sc_over_trials.png: shows the spike counts over trials
    """
    # Load data files
    rawsession_folder =  Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    kilosort_output_path = derivatives_base / "ephys" /  "concat_run"/"sorting"/ "sorter_output"
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    
    
    output_folder = derivatives_base/"analysis"/ "cell_characteristics"/ "unit_features"/ "spikecount_over_trials"
    output_folder.mkdir(exist_ok=True)


    # Here unit ids get filtered by unit_type
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    trial_length_df = get_trials_length_df(rawsession_folder)

   
    # Definitions
    bin_length = 30
    goal1_endtimes = get_goal_1_end_times(rawsession_folder, trials_to_include, last_trial_openfield = last_trial_openfield) if task == "hct" else None

            
    # Loading data
    print("Plotting spikecount over trials")
    print(f"Saving to {output_folder}")
    for unit_id in tqdm(unit_ids): 
        # output path for unit
        output_path = output_folder /  f"unit_{unit_id}_sc_over_trials.png"
        
        # loading spiketrain in seconds
        spike_train = get_spike_train_s(sorting, unit_id, sample_rate)

        # skip units with zero spikes
        if len(spike_train) == 0:
            continue
        
        # Getting number of bins
        total_trial_length = get_total_trial_length(trials_to_include, trial_length_df) # total length in s
        n_bins = total_trial_length/bin_length


        # Simulated adjacent trials
        trial_lengths = np.array(trial_length_df['trial length (s)'])
        trial_ends = np.cumsum(trial_lengths)
        trial_starts = np.concatenate(([0], trial_ends[:-1]))

        # Plot
        make_plot(spike_train, trial_starts, trial_ends, output_path, n_bins, unit_id, goal1_endtimes, last_trial_openfield = last_trial_openfield)
        
    print(f"Plots saved in {output_folder}")

if __name__ == "__main__":
    trials_to_include = np.arange(1,10)
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    plot_spikecount_over_trials(derivatives_base, trials_to_include)