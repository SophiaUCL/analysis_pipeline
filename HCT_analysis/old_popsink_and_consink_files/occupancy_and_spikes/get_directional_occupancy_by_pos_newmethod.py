import os
import numpy as np
from calculate_occupancy import get_axes_limits
import pandas as pd
from utilities.load_and_save_data import save_pickle, load_pickle
import spikeinterface.extractors as se
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
import json
from tqdm import tqdm

def get_xy_bins(limits, n_bins=100):

    # get the x and y limits of the maze
    x_min = limits['x_min']
    x_max = limits['x_max']
    x_width = limits['x_width']

    y_min = limits['y_min']
    y_max = limits['y_max']
    y_height = limits['y_height']

    # we want roughly 100 bins
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

def get_og_bins(bins):
   
    # get differences between each bin
    bin_diffs = np.diff(bins)

    # the last difference should be greater than the others, determine by how much
    diff = bin_diffs[-1] - np.mean(bin_diffs[:-1])

    # substract this difference from the last bin
    bins[-1] = bins[-1] - diff

    return bins



def get_directional_occupancy(directions, durations, n_bins=12):

    # create 12bins, each 24 degrees
    direction_bins_og = np.linspace(-np.pi, np.pi, n_bins+1)
    direction_bins = direction_bins_og.copy()
    direction_bins[0] = direction_bins_og[0] - 0.1 # subtract a small number from the first bin so that the first value is included in the bin
    direction_bins[-1] = direction_bins_og[-1] + 0.1 # add a small number to the last bin so that the last value is included in the bin

    # get the bin indices for each value in directions
    bin_indices = np.digitize(directions, direction_bins, right=True) - 1
    # any bin_indices that are -1 should be 0
    bin_indices[bin_indices==-1] = 0
    # any bin_indices that are n_bins should be n_bins-1
    bin_indices[bin_indices==n_bins] = n_bins-1

    # get the occupancy for each bin, so a vector of length n_bins
    occupancy = np.zeros(n_bins)
    for i in range(n_bins):
        occupancy[i] = np.sum(durations[bin_indices==i])
    return occupancy, direction_bins_og

def get_directional_occupancy_by_position(dlc_data, limits, frame_rate=25, n_dir_bins = 12):

    # NOTE THAT IN THE OUTPUT, THE FIRST INDEX IS THE Y AXIS, 
    # AND THE SECOND INDEX IS THE X AXIS

    # Returns the x and y coordinates for the bins
    x_bins, y_bins = get_xy_bins(limits, n_bins=100)

    # This seems to remove the last bit of the final element, but actually
    # (x_bins == x_bins_og).all() is True (and same for y array), so they
    # seem to be the same
    x_bins_og = get_og_bins(x_bins)
    y_bins_og = get_og_bins(y_bins)

    # create positional occupancy matrix
    n_x_bins = len(x_bins) - 1
    n_y_bins = len(y_bins) - 1

    # create directional occupancy by position array
    directional_occupancy_temp = np.zeros((n_y_bins, n_x_bins, n_dir_bins))

    # get x and y data 
    x = dlc_data['x']
    y = dlc_data['y']

    # NOTE: nan values are returned as the max bin here. 
    # We don't remove them here, but we do for the hd later, so there shouldn't be an error
    x_bin = np.digitize(x, x_bins) - 1
    x_bin[x_bin == n_x_bins] = n_x_bins - 1
    y_bin = np.digitize(y, y_bins) - 1 
    y_bin[y_bin == n_y_bins] = n_y_bins - 1

    # get the head direction
    hd = dlc_data['hd']

    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            # get the head directions and durations for these indices
            hd_temp = hd[indices]
            hd_temp = hd_temp[~np.isnan(hd_temp)] # Removing nan values, very important!
            
            # Jake had seperate durations for each value, we don't. The duration of one unit is constant
            # (one frame, thus 1/frame_rate seconds)
            durations_temp = np.ones(len(hd_temp))/frame_rate

            # get the directional occupancy for these indices
            directional_occupancy, direction_bins = \
                get_directional_occupancy(hd_temp, durations_temp, n_bins=n_dir_bins)

            # add the directional occupancy to positional_occupancy_temp
            directional_occupancy_temp[j, i, :] = directional_occupancy

    directional_occupancy_temp = np.round(directional_occupancy_temp, 3)

    directional_occupancy_by_position = {'occupancy': directional_occupancy_temp, 
                            'x_bins': x_bins_og, 'y_bins': y_bins_og, 'direction_bins': direction_bins}
    return directional_occupancy_by_position

def bin_spikes_by_position_and_direction_individual_units(unit_ids, sorting, rawsession_folder, directional_occupancy_by_position, pos_data, frame_rate = 25, sample_rate = 30000, goal = -1):
    
    # get the x and y bins
    x_bins_og = directional_occupancy_by_position['x_bins']
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-1 # add a small number to the last bin so that the last value is included in the bin

    y_bins_og = directional_occupancy_by_position['y_bins']
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-1 # add a small number to the last bin so that the last value is included in the bin

    # get the direction bins
    direction_bins_og = directional_occupancy_by_position['direction_bins']
    direction_bins = direction_bins_og.copy()
    direction_bins[0] = direction_bins_og[0] - 0.1 # subtract a small number from the first bin so that the first value is included in the bin
    direction_bins[-1] = direction_bins_og[-1] + 0.1 # add a small number to the last bin so that the last value is included in the bin

    n_bins = len(direction_bins) - 1

    x_org = pos_data.iloc[:, 0].to_numpy()
    y_org = pos_data.iloc[:, 1].to_numpy()
    hd_org = pos_data.iloc[:, 2].to_numpy()

    if np.nanmax(hd_org) > 2*np.pi + 0.1:
        hd_org = np.deg2rad(hd_org)
    
    
    # loop through the units
    spike_rates_by_position_and_direction = {'units': {}, 'x_bins': x_bins_og, 
                    'y_bins': y_bins_og, 'direction_bins': direction_bins_og}

    
    for u in tqdm(unit_ids):
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=u)
        spike_train_secs = spike_train_unscaled / sample_rate
        if goal > 0:
            # restrict if goal is defined
            spike_train_secs_g = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder, goal=goal)
        else:
            spike_train_secs_g = spike_train_secs
        spike_train = np.round(spike_train_secs_g*frame_rate) # trial data is now in frames in order to match it with xy data
        spike_train =  np.array([np.int32(el) for el in spike_train if el < len(pos_data)]) # Ensure spike train is within bounds of x and y

        if len(spike_train) == 0:
            continue
        # Finding spike times for this unit
        x = x_org[spike_train]
        y = y_org[spike_train]
        hd = hd_org[spike_train]

        # We're removing nan values here, otherwise they get put in the last bin and that gives errors
        mask = np.isnan(hd)
        x = x[~mask]
        y = y[~mask]
        hd = hd[~mask]
        spike_rates_by_position_and_direction['units'][u] = \
            np.zeros((len(y_bins)-1, len(x_bins)-1, n_bins))
        
        # Get x and y as spike times of unit

        if x.size == 0:
            continue
            
        # sort the spike positions into bins
        x_bin = np.digitize(x, x_bins) - 1
        y_bin = np.digitize(y, y_bins) - 1


        for i in range(np.max(x_bin)+1):
            for j in range(np.max(y_bin)+1):

                # get the directional occupancy for x_bin == i and y_bin == j
                directional_occupancy = directional_occupancy_by_position['occupancy'][j, i]

                # get the indices where x_bin == i and y_bin == j
                indices = np.where((x_bin == i) & (y_bin == j))[0]

                # get the head directions for these indices
                hd_temp = hd[indices]

                # sort the head directions into bins
                bin_indices = np.digitize(hd_temp, direction_bins, right=True) - 1
                bin_indices[bin_indices==-1] = 0
                # any bin_indices that are n_bins should be n_bins-1
                bin_indices[bin_indices==n_bins] = n_bins-1

                # get the spike counts and rates
                spike_counts_temp = np.zeros(n_bins)
                spike_rates_temp = np.zeros(n_bins)

                for b in range(n_bins):
                    # get the spike counts for the current bin
                    spike_counts_temp[b] = np.sum(bin_indices==b)

                    
                    if directional_occupancy[b] == 0:
                        spike_rates_temp[b] = 0.
                    else:
                        # divide the spike counts by the occupancy
                        spike_rates_temp[b] = np.round(spike_counts_temp[b] / directional_occupancy[b], 3)

                # place the spike rates in the correct position in the array
                spike_rates_by_position_and_direction['units'][u][j, i, :] = spike_rates_temp

    return spike_rates_by_position_and_direction

def get_limits_from_json(derivatives_base):
    """Gets the xy limits from the json file created in the get_limits.py function"""
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    return limits["xmin"], limits["xmax"], limits["ymin"], limits["ymax"]

def main(derivatives_base, rawsession_folder,  code_to_run = []):
    """
    Main function to calculate directional occupancy by position and spike rates by position and direction.
    NOTE currently does this for all cells, not just good ones
    Args:
        derivatives_base (str): The base directory for the derivatives.
        rawsession_folder (str): The folder containing the raw session data.
        code_to_run (list, optional): List of codes to run. Defaults to [].
        
    """
    # Used to calculate spike_rates_by_position_and_direction
    x_min, x_max, y_min, y_max = get_limits_from_json(derivatives_base)

    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])

    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids


    # get x and y limits
    limits = get_axes_limits(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'spatial_features', 'consink_data_newmethod')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if 0 in code_to_run:
        print("Getting directional occupancy by position")
        directional_occupancy_by_position = get_directional_occupancy_by_position(pos_data, limits)
        save_pickle(directional_occupancy_by_position, 'directional_occupancy_by_position', output_folder)
        for g in [0, 1,2]:
            # This should work
            pos_data_path_goal = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{g}_trials.csv')
            pos_data_goal = pd.read_csv(pos_data_path_goal)
            if np.nanmax(pos_data_goal['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
                pos_data_goal['hd'] = np.deg2rad(pos_data_goal['hd'])
            print(f"Getting directional occupancy by position for goal {g}")
            directional_occupancy_by_position = get_directional_occupancy_by_position(pos_data_goal, limits)
            save_pickle(directional_occupancy_by_position, f'directional_occupancy_by_position_g{g}', output_folder)
        print(f"Pickle saved to {output_folder }")
    
    if 1 in code_to_run:
        # load the directional occupancy by position data
        directional_occupancy_by_position = load_pickle('directional_occupancy_by_position', output_folder)
        # bin spikes by position and direction
        print("Binning spikes for full trials")
        spike_rates_by_position_and_direction = bin_spikes_by_position_and_direction_individual_units(unit_ids, sorting, rawsession_folder,
                                                directional_occupancy_by_position, pos_data)

        # save the spike rates by position and direction
        save_pickle(spike_rates_by_position_and_direction, 'spike_rates_by_position_and_direction', output_folder)
        
        for g in [0, 1,2]:
            print(f"Binning spikes for goal {g}")
            print(f"Getting directional occupancy by position for goal {g}")

            # CHANGED HERE, NOW USING POS DATA INSTEAD OF POS DATA GOAL
            directional_occupancy_by_position_goal = load_pickle(f'directional_occupancy_by_position_g{g}', output_folder)

            spike_rates_by_position_and_direction_goal = bin_spikes_by_position_and_direction_individual_units(unit_ids, sorting, rawsession_folder,
                                                    directional_occupancy_by_position_goal, pos_data, goal=g)
            # save the spike rates by position and direction
            save_pickle(spike_rates_by_position_and_direction_goal, f'spike_rates_by_position_and_direction_g{g}', output_folder)
            



    
if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    main(derivatives_base, rawsession_folder, [0,1])