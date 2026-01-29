import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.widgets as sw
import probeinterface
import os
import time
from scipy.io import savemat


from spikeinterface.qualitymetrics import (
    compute_snrs,
    compute_firing_rates,
    compute_isi_violations,
    calculate_pc_metrics,
    compute_quality_metrics,
)
from spikeinterface.comparison import compare_sorter_to_ground_truth
from probeinterface import Probe, ProbeGroup
from pathlib import Path
from tqdm import tqdm

def create_waveform_df(sorting_analyzer, unit_ids, labels, unit_features_path):
    """
    Creates df with waveform metrics
    """
    output_folder = os.path.join(unit_features_path,"all_units_overview")
    output_path = os.path.join(output_folder, "unit_waveform_metrics.csv")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    tm = sorting_analyzer.compute(input="template_metrics")
    df = pd.DataFrame(tm.data['metrics'])
    df.insert(0, 'unit_ids', unit_ids)
    df.insert(1, 'label', labels)
    df.to_csv(output_path, index=False)

def create_metrics_df(sorting_analyzer, unit_ids, labels, unit_features_path):
    """ Creates df with unit metrics. Saves to unit_features/all_units_overview"""
    output_folder = os.path.join(unit_features_path,"all_units_overview")
    output_path_df = os.path.join(output_folder, "unit_metrics.csv")
    
    # Removed amplitude cutoff because it gave an error
    #metrics_v2 = compute_quality_metrics(sorting_analyzer, metric_names=["firing_rate", "snr", "amplitude_cutoff", "isi_violation"])
    metrics_v2 = compute_quality_metrics(sorting_analyzer, metric_names=["firing_rate", "snr", "isi_violation"])
    metrics_v2.insert(0, 'unit_ids', unit_ids)
    metrics_v2.insert(1, 'label', labels)
    
    #desired_columns = ['unit_ids', 'label', 'firing_rate', 'snr', 'amplitude_cutoff', 'isi_violations_ratio', 'isi_violations_count']
    desired_columns = ['unit_ids', 'label', 'firing_rate', 'snr', 'isi_violations_ratio', 'isi_violations_count']
    df = metrics_v2[desired_columns]
    df.to_csv(output_path_df, index=False)
    return df
def get_recording_data(spikeinterface_recording_path):
    """ 
    Gets gain_to_uV and offset_to_uV from the data file (which contains recording data)
    
    The path to the data for some reason keeps changing in different versions of spikeinterface
    If this format doesn't work, feed the file into chatGPT and then ask it to find the path. 
    Note that gain_to_uV and offset_to_uV are lists, so we take the first element 
    """
    # Obtain gain_to_uV and offset value
    with open(spikeinterface_recording_path, "r") as f:
        data = json.load(f)
        
    try:
        # Format option 1
        gain_to_uV = data["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording_list"][0]["properties"]["gain_to_uV"][0]
        offset_to_uV = data["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording_list"][0]["properties"]["offset_to_uV"][0]
    except (KeyError, IndexError, TypeError):
        try: # format option 2
            parent = data['kwargs']['parent_recording']
            properties = parent['properties']
            gain_to_uV = properties['gain_to_uV'][0]
            offset_to_uV = properties['offset_to_uV'][0]
        except (KeyError, IndexError, TypeError):
            # Format option 3
            gain_to_uV = data["properties"]["gain_to_uV"][0]
            offset_to_uV = data["properties"]["offset_to_uV"][0]
    
    print("gain_to_uV:", gain_to_uV)
    print("offset_to_uV:", offset_to_uV)
    
    return gain_to_uV, offset_to_uV

def load_recording(recording_path, probe_path, sample_rate, gain_to_uV, offset_to_uV, kilosort_output_path):
    print("Loading recording")
    recording = se.read_binary(
        file_paths = recording_path,
        # Info below is found in the json file in the same folder
        sampling_frequency=sample_rate,
        dtype = np.int16,  
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
        num_channels = 384,
    )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )

    unit_ids = sorting.unit_ids
    labels = sorting.get_property('KSLabel')
    good_units_ids = [el for el in unit_ids if labels[el] == 'good']
    colour_scheme = ['blue' if labels[el] == 'good' else 'red' for el in unit_ids]
    
    # Loading probe data. Assumes one probe
    probe_group = probeinterface.read_probeinterface(probe_path)
    probe = probe_group.probes[0]   
    recording = recording.set_probe(probe)
    
    return recording, sorting, unit_ids, good_units_ids, labels, colour_scheme
    
def save_trial_duration(recording, derivatives_base):
    """ Obtains the length of the recording, prints it and saves it to analysis/metadata/trialduration"""
    total_samples = recording.get_num_frames()
    sampling_rate = recording.get_sampling_frequency()
    total_duration_sec = total_samples / sampling_rate
    formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_duration_sec))
    print(f"Total trial length: {formatted_time}")
    
    time_output_folder =  os.path.join(derivatives_base, "analysis", "metadata")
    if not os.path.exists(time_output_folder):
        os.makedirs(time_output_folder)
    time_output_path = os.path.join(time_output_folder, "trialduration.txt")

    with open(time_output_path, "w") as f:
        f.write('%f' % total_duration_sec)

def get_sorting_analyzer(run_analyzer_from_memory, analyzer_path, sorting, recording):
    """ Creates sorting analyzer or loads it from memory"""
    # Run analyzer from memory or create new one
    if run_analyzer_from_memory:
        print("Loading sorting analyzer")
        sorting_analyzer = si.load_sorting_analyzer(
            folder = analyzer_path
        )
    else:
        print("Creating  sorting analyzer")
        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            folder=analyzer_path,
            format = "binary_folder",
            overwrite = True,
        )
        sorting_analyzer.compute(["correlograms", "random_spikes", "waveforms", "templates", "noise_levels"], save=True)
        sorting_analyzer.compute(["spike_amplitudes", "unit_locations", "spike_locations"], save = True)
    return sorting_analyzer

def plot_unit_presence(sorting, unit_features_path):
    """ Plots unit presence over time"""
    # plotting unit presence
    widget = sw.plot_unit_presence(sorting) 
    output_path = os.path.join(unit_features_path,"all_units_overview", "unit_presence_all_units.png")
    fig = widget.figure
    fig.suptitle("Unit presence over time (all units)")
    plt.savefig(output_path)
    plt.close(fig)

def plot_unit_location(unit_features_path, sorting_analyzer, good_units_ids = None, colour_scheme = None):
    """ Plots unit location of all units or good units"""
    if good_units_ids is None:
        output_path = os.path.join(unit_features_path,"all_units_overview", "unit_location_all_units.png")
        title = "Unit location (all units)"
        widget = sw.plot_unit_locations(sorting_analyzer, unit_colors = colour_scheme, figsize=(8, 16))
    else:
        output_path = os.path.join(unit_features_path,"all_units_overview", "unit_location_all_units.png")
        title = "Unit location (good units)"
        widget = sw.plot_unit_locations(sorting_analyzer, unit_ids = good_units_ids,  figsize=(8, 16))
   
    fig = widget.figure
    ax = fig.axes[0]
    ax.set_xlim(-200, 200)     
    ax.set_ylim(0, 4000)      
    ax.set_title(title)
    plt.savefig(output_path)
    plt.close(fig)

def plot_unit_depth(unit_features_path, sorting_analyzer, colour_scheme):
    """ Plots unit depth for all units"""
    output_path = os.path.join(unit_features_path,"all_units_overview", "unit_depth_all_units.png")
    widget = sw.plot_unit_depths(sorting_analyzer, unit_colors = colour_scheme, figsize=(8, 16))
    fig = widget.figure
    ax = fig.axes[0]
    ax.set_xlim(-200, 200)      
    ax.set_ylim(0, 4000)       
    ax.set_title("Unit depth")
    plt.savefig(output_path)
    plt.close(fig)

def load_metrics_df(unit_features_path):
    """ Returns unit_metrics.csv df"""
    output_folder = os.path.join(unit_features_path,"all_units_overview")
    output_path_df = os.path.join(output_folder, "unit_metrics.csv")
    df = pd.read_csv(output_path_df)
    return df

def get_unit_info(labels, sample_rate, df, unit_features_path, unit_id, sorting):
    """ Obtains info needed to make plots"""
    label = labels[unit_id]
    
    output_dir = os.path.join(unit_features_path, "auto_and_wv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Obtaining cell info
    firing_rate = df['firing_rate'].iloc[unit_id]

    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    if len(spike_train_unscaled) > 10000:
        spike_train_unscaled = spike_train_unscaled[:int(1e4)]
        
    spike_train = np.float64(spike_train_unscaled/sample_rate)
    return  label, firing_rate, spike_train, output_dir
   
def make_autocorrelogram(spike_train, width, ax):
    """ Creates autocorrelogram of width = width
    
    Inputs:
        spike_train: spike train in seconds
        width: width of autocorrelogram (10 or 500)
        ax: plot
    """
    bins, cou = interspike_histogram(spike_train, spike_train, width)
    if width == 10:
        title = f'{np.sum(cou[51:60])}/{len(spike_train)}'
        lim = 11
    elif width == 500:
        title = 'Autocorrelogram (500 ms window)'
        lim = 550
    else:
        raise ValueError("Error: select correct autocorrelogram width (10 or 500ms)")
    binwidth = bins[1] - bins[0]
    ax.bar(bins, cou, width=binwidth, align='center')
    ax.set_title(title)
    ax.set_xlim([-lim, lim])    


def interspike_histogram(spkTr1, spkTr2, maxInt):
    """
    Calculate the interspike histogram between two spike trains.
    Python version of Marius' code

    Inputs:
    spkTr1: Spike train 1 
    spkTr2: Spike train 2
    maxInt: Maximum interval for histogram in ms

    Outputs:
    bin_centers: Centers of the histogram bins
    counts: Counts of spikes in each bin
    """
    # Convert to ms
    spkTr1 = spkTr1 * 1000
    spkTr2 = spkTr2 * 1000
    n_divisions = 50 # Default

    # Finding intervals
    nSpk = len(spkTr1)

    int_matrix = np.zeros((nSpk, nSpk - 1))

    for ii in range(1, nSpk):
        # Shifting spkTr2 by ii positions
        shifted = np.roll(spkTr2, ii)
        # Fill column ii-1 with element-wise difference
        int_matrix[:, ii - 1] = spkTr1 - shifted

    binwidth = maxInt / n_divisions
    bins = np.arange(-maxInt, maxInt + binwidth, binwidth)
    if nSpk == 0 or len(spkTr2) == 0:
        # Failure for when 0 spikes
        counts = np.full_like(bins, np.nan)
    else:
        # Otherwise
        counts, _ = np.histogram(int_matrix.flatten(), bins=bins)

    bin_centers = bins[:-1] + binwidth / 2
    return bin_centers, counts
   

def plot_waveform(sorting_analyzer, unit_id, ax):
    """ Makes plot with the waveform channels for unit_id"""
    waveforms_ext = sorting_analyzer.get_extension("waveforms")
    wf = waveforms_ext.get_waveforms_one_unit(unit_id=unit_id)
    mean_wf = wf.mean(axis=0)

    max_range = 0
    max_channel = 0
    for ch in range(mean_wf.shape[1]):
        data = mean_wf[:,ch]
        range_val = np.max(data) - np.min(data)
        if range_val > max_range:
            max_range = range_val
            max_channel = ch
    
    min_channel_range = np.max([0, max_channel - 4])
    max_channel_range = np.min([mean_wf.shape[1], max_channel + 4])

    for ch in range(min_channel_range, max_channel_range):
        ax.plot(mean_wf[:, ch], label=f'Ch {ch}')

    ax.set_title(f'Unit {unit_id} Mean Waveform')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.legend(loc = 'center right') 

def load_paths(derivatives_base):
    """ Loads all the needed paths"""
    print("=== Running feature extraction in Spikeinterface ===")
    recording_path = os.path.join(derivatives_base, "ephys",  "concat_run", "preprocessed", "traces_cached_seg0.raw")
    probe_path =  os.path.join(derivatives_base, "ephys","concat_run", "preprocessed", "probe.json")
    kilosort_output_path = os.path.join(derivatives_base,"ephys", "concat_run","sorting", "sorter_output" )
    unit_features_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features")
    spikeinterface_recording_path = os.path.join(derivatives_base, "ephys","concat_run","sorting", "spikeinterface_recording.json" )
    kilosort_output_path = Path(kilosort_output_path)
    if not os.path.exists(unit_features_path):
        os.makedirs(unit_features_path)

    analyzer_path = os.path.join(unit_features_path, "spikeinterface_data")
    if not os.path.exists(analyzer_path):
        os.makedirs(analyzer_path)
    return recording_path, probe_path, kilosort_output_path, unit_features_path, spikeinterface_recording_path, analyzer_path
         
def run_spikeinterface(derivatives_base, run_analyzer_from_memory = False, run_df_from_memory = False,  sample_rate = 30000):
    """
    Runs spikeinterface on the processed data. 

    Inputs:
        derivatives_base: path to derivatives folder
        run_analyzer_from_memory: shows whether to run the analyzer from memory or create a new one (default = False)
        run_analyzer_from_memory: whether to create the two dfs with unit features or run from memory (default = FAlse)
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
        

    print("Progress plotting waveforms and autocorrelograms for all cells")
    for unit_id in tqdm(unit_ids):
        label, firing_rate, spike_train, output_dir = get_unit_info(labels, sample_rate, df, unit_features_path, unit_id, sorting)

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
    