
import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from unit_features.utils import load_unit_ids, load_trial_xpos, get_spikes_tr, plot_trial_firing, load_directories, get_unit_info, get_trial_length_info
UnitTypes = Literal['pyramidal', 'good', 'all']



def plot_firing_each_epoch(derivatives_base: Path, trials_to_include: list, unit_type: UnitTypes, frame_rate: int = 25, sample_rate: int = 30000):
    """For each unit, creates an n by 3 plot showing the firing rate for each trial 
    with each epoch imdicated. 

    Args:
        derivatives_base (Path): path to derivatives folder_
        trials_to_include (list): array with trial numbers.
        unit_type: type of units to visualize for (pyramidal, good, or all)
        frame_rate (int, optional): Frame rate of video. Defaults to 25.
        sample_rate (int, optional): Sample rate of recording. Defaults to 30000.

    Raises:
        FileNotFoundError: _description_
        
    Outputs:
        derivatives_base /analysis/cell_characteristics/unit_features/epochs_spike_count/unit_{unit_id}_epoch_firing.png
        firing of each unit for each epoch in each trial
    """
    # Loading rawsession folder
    rawsession_folder =  Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    kilosort_output_path = derivatives_base / "ephys" /  "concat_run"/"sorting"/ "sorter_output"
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)

    # Loading data
    pos_data_dir, epoch_times, df_unit_metrics, trials_length, output_dir = load_directories(derivatives_base, rawsession_folder)
    
    # Trial information    
    n_trials = len(trials_to_include)
    n_epochs = 3
    
    n_cols = 3
    n_rows = np.ceil(n_trials/n_cols).astype(int)
    trial_dur_so_far = 0

    print(f"Plotting firing for each epoch. Saving to {output_dir}")
    for unit_id in tqdm(unit_ids):
        # Getting spiketrian for unit
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled/sample_rate) # Now its in seconds

        # Unit information
        unit_firing_rate, unit_label = get_unit_info(df_unit_metrics, unit_id)

        # Creating figure
        fig, axs = plt.subplots(n_rows, n_cols, figsize = [n_cols*7, n_rows*5])
        fig.suptitle(f"Unit {unit_id}. Label: {unit_label}. Firing rate = {unit_firing_rate:.1f} Hz", fontsize = 18)
        axs = axs.flatten()
        counter = 0 # counts which axs we're on
        trial_dur_so_far = 0
        
        # Looping over trials
        for tr in trials_to_include:
            # Trial times
            start_time, trial_length, trial_row = get_trial_length_info(epoch_times, trials_length,  tr)
            
            # Get spiketrain for this trial
            x = load_trial_xpos(pos_data_dir, tr)
            spike_train_this_trial = get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate)
            
            # Make plots
            plot_trial_firing(spike_train_this_trial,trial_row, n_epochs, tr, x, frame_rate, ax = axs[counter])
            
            counter += 1
            trial_dur_so_far += trial_length


        output_path = output_dir/ f"unit_{unit_id}_epoch_firing.png"
        plt.savefig(output_path)
        plt.close(fig)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1,11)
    plot_firing_each_epoch(derivatives_base, rawsession_folder, trials_to_include)