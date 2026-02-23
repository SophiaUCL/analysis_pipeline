import matplotlib.pyplot as plt
import os
from pathlib import Path
from tqdm import tqdm

from preprocessing.spikeinterface_utils import get_recording_data, load_recording, get_sorting_analyzer, plot_unit_presence, get_unit_info, save_trial_duration, load_metrics_df, create_metrics_df, create_waveform_df,  plot_waveform, make_autocorrelogram, load_paths
import shutil

def run_spikeinterface(derivatives_base: Path, run_analyzer_from_memory: bool = False, run_df_from_memory: bool = False, clear_plot_folder: bool = False, sample_rate: int = 30000):
    """
    Runs spikeinterface on the processed data. 

    Inputs:
        derivatives_base: path to derivatives folder
        run_analyzer_from_memory: shows whether to run the analyzer from memory or create a new one (default = False)
        run_analyzer_from_memory: whether to create the two dfs with unit features or run from memory (default = FAlse)
        clear_plot_folder (bool: False): Whether to clear the auto_wv folder before putting plots in it. Set to True after merging units.
        sample_rate: sampling rate of recording (default = 30000)

    Saves:
    df with waveform metrics
    df with other unit metrics (firing_rate', 'snr', 'isi_violations_ratio', 'isi_violations_count')
    plots with autocorrelogram and waveforms in analysis/unit_features/auto_and_wv
    plots with information about the units in analysis/unit_features/all_units_overview
    
    """
    # Loading paths 
    recording_path, probe_path, kilosort_output_path, unit_features_path, spikeinterface_recording_path, analyzer_path = load_paths(derivatives_base)
    gain_to_uV, offset_to_uV = get_recording_data(spikeinterface_recording_path)

    # Loading recording data
    recording, sorting, unit_ids, good_units_ids, labels, colour_scheme = load_recording(recording_path, probe_path, sample_rate, gain_to_uV, offset_to_uV, kilosort_output_path)
    
    
    # Obtaining trial duration
    save_trial_duration(recording, derivatives_base)
    # Getting the sorting analyser
    sorting_analyzer = get_sorting_analyzer(run_analyzer_from_memory, analyzer_path, sorting, recording)


    if not run_df_from_memory:
        print("Creating dataframes and plots")
        # Creating dfs
        create_waveform_df(sorting_analyzer, unit_ids, labels, unit_features_path)
        df = create_metrics_df(sorting_analyzer, unit_ids, labels, unit_features_path)
        
        # Plotting
        plot_unit_presence(sorting, unit_features_path)
        #plot_unit_location(unit_features_path, sorting_analyzer, colour_scheme = colour_scheme) # all units
        #plot_unit_location(unit_features_path, sorting_analyzer, good_units_ids= good_units_ids) # good units
        #plot_unit_depth(unit_features_path, sorting_analyzer, colour_scheme)

    else:
        df = load_metrics_df(unit_features_path)
        

    output_dir = unit_features_path / "auto_and_wv"
    if clear_plot_folder:
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

        
    print("Progress plotting waveforms and autocorrelograms for all cells")
    print(f"Saving to {output_dir}")
    for unit_id in tqdm(unit_ids):
        label, firing_rate, spike_train = get_unit_info(df, unit_id, sorting, sample_rate)

        if isinstance(spike_train,  (int, float)):
            print(f"Unit {unit_id} has 0 or 1 spikes, skipping")
            continue

        # Making plot
        fig, axs = plt.subplots(3, 1, figsize=(8, 10)) 
        fig.suptitle(f'Unit {unit_id}, label: {label}, firing rate: {firing_rate:.1f} Hz')

        # 1. Autocorrelogram 10 ms window
        make_autocorrelogram(spike_train, width = 10, ax = axs[0])

        # 2. Autocorrelogram 500 ms window
        make_autocorrelogram(spike_train, width = 500, ax = axs[1])

        # 3. Waveform
        plot_waveform(sorting_analyzer, unit_id, ax = axs[2])

        filename = f'unit_{unit_id:03d}.png'
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close(fig)
        
    print("Spikeinterface tasks completed")
    



if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    run_spikeinterface(derivatives_base, True, True)
    