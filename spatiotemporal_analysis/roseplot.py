import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from spatiotemporal_analysis.utils.spatial_features_plots import plot_roseplots, add_arm_overlay_roseplot
from spatiotemporal_analysis.utils import get_MRL_data, get_directories, get_sum_bin

                     
def make_roseplots(derivatives_base: Path, trials_to_include: list, deg: int, path_to_df: Path = None) -> None:
    """
    This code creates the roseplots as visualized for the spatiotemporal analysis. 
    For the location of the arms (N, NE, etc.) is assumes we're using the new camera (the one with 25 frame rate and north on top)

    Input: 
    derivatives_base; path to derivatives folder
    trials_to_include: trials to include in the analysis
    deg: degree that the data was binned into
    path_to_df: path to df containing MRl data (optional)
    """
    
    # Getting df with MRL values for each unit and epoch
    df_all = get_MRL_data(derivatives_base, path_to_df)
    df = df_all[df_all['significant'] == 'sig']

    # Dataframe with raised arms
    behaviour_df, output_path_plot = get_directories(derivatives_base, deg)

    # Direction of arms and their angles
    arms_dir = ["N", "NW", "SW", "S", "SE", "NE"]
    arms_angles_start = [30, 90, 150, 210, 270, 330]
    # Plot: 3 columns for epochs, final column for correct/incorrect
    fig, axs = plt.subplots(len(trials_to_include), 4, figsize = [3*4, 4*len(trials_to_include)], subplot_kw = {'projection': 'polar'})
    num_bins = 24

    for tr in trials_to_include:
        for e in np.arange(1,4):
            
            num_spikes_arr = []
            mean_dir_arr = []


            # filter for this trial and epoch
            filtered_df = df[(df['trial'] == tr) & (df['epoch'] == e)]

            # if any cells were significant this trial epoch
            if len(filtered_df) > 0:
                mean_dir_arr = np.array(filtered_df['mean_direction'])
                num_spikes_arr = np.array(filtered_df['num_spikes'])

                # Binning the data
                counts, bin_edges = np.histogram(mean_dir_arr, bins = num_bins, range = (-180, 180))

                # Finding the bin of each element in the mean dir arr
                sum_count_bin = get_sum_bin(mean_dir_arr, num_spikes_arr, bin_edges)
                
                plot_roseplots(filtered_df,behaviour_df, arms_dir, arms_angles_start, sum_count_bin, bin_edges,e,  tr, axs[tr-1, e-1])

            
            axs[tr-1, e-1].set_title(f" Tr {tr} epoch {e}")

        add_arm_overlay_roseplot(behaviour_df, tr, trials_to_include,  axs[tr-1, 3], fig)

    plt.tight_layout()
    plt.savefig(output_path_plot)
    plt.show()
    print(f"Saved figure to {output_path_plot}")

if __name__ == "__main__":
    trials_to_include = np.arange(1,6)
    print(trials_to_include)
    derivatives_base = r"S:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-05_date-18072025\rerun_1212"
    make_roseplots(derivatives_base, trials_to_include, 15)
