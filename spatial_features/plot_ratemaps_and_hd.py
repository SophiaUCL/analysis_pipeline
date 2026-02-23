import numpy as np
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
import shutil
from astropy.stats import circmean
from spatial_features.utils.spatial_features_utils import load_unit_ids, get_outline, get_limits, get_posdata, get_occupancy_time, get_ratemaps, get_spike_train_frames, get_directional_firingrate
from spatial_features.utils.spatial_features_plots import plot_rmap, plot_occupancy, plot_directional_firingrate
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap, ListedColormap

cmap_Cristina= ListedColormap(["#384682", "#578da1", "#62b378", "#e6dc77", '#f0754d'])

UnitTypes = Literal['pyramidal', 'good', 'all']

def plot_ratemaps_and_hd(derivatives_base: Path, unit_type: UnitTypes, save_plots: bool =True, show_plots: bool = False, clear_plot_folder: bool = False, frame_rate: int = 25, sample_rate: int = 30000) -> None:
    """ 
    For each unit, makes a plot for each unit with its ratemap (left), occupancy (centre) and directional firing rate (right)

    Inputs
    -------
    derivatives_base (Path): Path to derivatives folder
    unit_type (pyramidal, good, or all): units for which the plots will be made
    frame_rate (int: 25): frame rate of camera
    sample_rate (int: 30000): sample rate of recording
    save_plots (bool: True): whether plots are saved to the folder
    show_plots (bool: False): whether plots are displayed
    clear_plot_folder (bool: False): whether to clear all the plots in the plot folder (set to True after merging)
    
    Saves
    --------
    derivatives_base/'analysis'/ 'cell_characteristics'/'spatial_features'/ 'ratemaps_and_hd'/f"unit_{unit_id}_rm_hd.png" - rmap and hd for each unit
    
    Called by
    ---------
    spatial_processing_pipeline.py
    """
    # Load data files
    kilosort_output_path = derivatives_base/ 'ephys'/ "concat_run"/"sorting"/ "sorter_output" 
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    
    # Output folder
    output_folder = derivatives_base/'analysis'/ 'cell_characteristics'/'spatial_features'/ 'ratemaps_and_hd'
    if clear_plot_folder:
        print("Clearing output folder")
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents= True, exist_ok = True)
    
    
    # Here unit ids get filtered by unit_type
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    
    # Limits and outline
    xmin, xmax, ymin, ymax = get_limits(derivatives_base)
    outline_x, outline_y = get_outline(derivatives_base)
        
    # Get directory for the positional data
    x, y, hd,_ = get_posdata(derivatives_base, method = "ears")

    # Obtaining hd for this trial how much the animal sampled in each bin
    num_bins = 24
    occupancy_time = get_occupancy_time(hd, frame_rate, num_bins = num_bins)

     # Loop over units
    print("Plotting ratemaps and hd")
    print(f"Saving plots to {output_folder}")
    for unit_id in tqdm(unit_ids):
        
        # Load spike data
        spike_train = get_spike_train_frames(sorting, unit_id, x, sample_rate, frame_rate)
        
        # Make plot
        fig, axs = plt.subplots(1, 3, figsize = [15, 5])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}", fontsize = 18)

        # ===== Plot ratemap ====
        rmap, x_edges, y_edges= get_ratemaps(spike_train, x, y, 3, binsize=36, stddev=25)

        
        plot_rmap(rmap, xmin, xmax, ymin, ymax, x_edges, y_edges, outline_x, outline_y, ax = axs[0], fig = fig, title = f"n:{len(spike_train)}")
       
        # ==== Plot occupancy ====
        plot_occupancy(x, y, xmin, xmax, ymin, ymax, outline_x, outline_y, frame_rate, axs[1], fig)

        # === Plot HD ==
        direction_firing_rate, bin_centers = get_directional_firingrate(hd, spike_train, num_bins, occupancy_time)
        fig.delaxes(axs[2])
        axs[2] = fig.add_subplot(1,3,3, polar=True)
        plot_directional_firingrate(bin_centers, direction_firing_rate, ax = axs[2])

        output_path = output_folder /  f"unit_{unit_id}_rm_hd.png"
        if save_plots:
            plt.savefig(output_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    
    print(f"Saved plots to {output_folder}")
         

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    plot_ratemaps_and_hd(derivatives_base,unit_type = "good", save_plots=False, show_plots=True)



