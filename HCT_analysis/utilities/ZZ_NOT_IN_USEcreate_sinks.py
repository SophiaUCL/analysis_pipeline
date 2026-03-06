import numpy as np
import os
import json
from utilities.load_and_save_data import save_pickle
from get_limits import plot_sink_bins

""" NOT IN USE"""
def get_xy_bins(limits, n_bins=120):

    # get the x and y limits of the maze
    x_min = limits['x_min']
    x_max = limits['x_max']
    x_width = limits['x_width']

    y_min = limits['y_min']
    y_max = limits['y_max']
    y_height = limits['y_height']

    # we want roughly 120 bins
    pixels_per_bin = np.sqrt(x_width*y_height/n_bins)
    n_x_bins = int(np.round(x_width/pixels_per_bin)) # note that n_bins is actually one more than the number of bins
    n_y_bins = int(np.round(y_height/pixels_per_bin))

    # create bins
    x_bins_og = np.linspace(x_min, x_max, n_x_bins + 1)
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin
    
    y_bins_og = np.linspace(y_min, y_max, n_y_bins + 1)
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin

    return x_bins, y_bins


def create_sinks_and_dirbins(derivatives_base, n_bins, folder_name):
    """ Creates sink bins
    Saves them as a pickle file in derivatives_base/analysis/spatial_features/{folder_name}/sink_bins.pkl
    and plots them"""
    if not os.path.exists(os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', folder_name)):
        os.makedirs(os.path.join(derivatives_base, 'analysis','cell_characteristics',  'spatial_features', folder_name))
        
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
        
    x_bins_og, y_bins_og = get_xy_bins(limits, n_bins=n_bins)


    bins = {'x': x_bins_og, 'y': y_bins_og}

    # sink position in centres
    x_sink_pos = x_bins_og[0:-1] + np.diff(x_bins_og)/2
    y_sink_pos = y_bins_og[0:-1] + np.diff(y_bins_og)/2

    candidate_sinks = {'x': x_sink_pos, 'y': y_sink_pos}
    
    plot_sink_bins(derivatives_base, sink_bins = candidate_sinks)
    save_pickle(candidate_sinks, 'sink_bins', os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', folder_name))
    
    direction_bins = np.linspace(-np.pi, np.pi,12+1)
    save_pickle(direction_bins, 'direction_bins', os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', folder_name))
    print(f"Pickles saved to {os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', folder_name)}")