
import numpy as np
import pandas as pd

""" Functions to calculate the relative directional occupancy (used to be in find_occupancy.py)"""

def get_relative_direction_occupancy_by_position_platformbins(
    pos_data: pd.DataFrame,
    sink_positions: list,  
    num_candidate_sinks: int= 127, 
    n_dir_bins: int=12, 
    frame_rate: int =25
    ):
    """
    Compute relative-direction occupancy per platform for all candidate sinks.

    For each candidate sink (n = 127) and each platform (n = 61),
    calculates the time spent in each relative-direction bin (n_dir_bins),
    where relative direction is defined as:
        direction_to_sink − head_direction.

    Returns
    -------
    reldir_occ_by_pos : np.ndarray
        Array of shape (num_candidate_sinks, 61, n_dir_bins)
        containing time occupancy (seconds) per direction bin.
    """

    # create relative directional occupancy by position array
    reldir_occ_by_pos = np.zeros((num_candidate_sinks, 61, n_dir_bins))

    # get x and y data
    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()
    platforms = pos_data['platform'].to_numpy()

    # Remove nan values, otherwise binning gets funky
    mask = np.isnan(hd) | np.isnan(x) | np.isnan(platforms)
    x = x[~mask]
    y = y[~mask]
    hd = hd[~mask]
    platforms = platforms[~mask]


    # Going over positional bins
    for p in range(61):
        # get the indices where x_bin == i and y_bin == j
        indices = np.where(platforms == p + 1)[0]
        if len(indices) == 0:
            continue
        try:
            # Get hd and positions for these frames
            x_positions = x[indices]
            y_positions = y[indices]
            hd_temp = hd[indices]

        except:
            print(p)
            breakpoint()
        positions = {'x': x_positions, 'y': y_positions}



        for s in range(num_candidate_sinks):
            # get directions to sink
            platform_loc = sink_positions[s]
            directions = get_directions_to_position([platform_loc[0], platform_loc[1]], positions)

            # get the relative direction
            relative_direction = get_relative_directions_to_position(directions, hd_temp)
            durations_temp = np.ones(
                len(relative_direction)) / frame_rate  # NOTE: Jake's code used durations data from DLC.
            # We don't use that, each frame is equally long, so we replace all durations with 1/frame_rate

            # get the directional occupancy for these indices
            directional_occupancy, direction_bins = \
                get_directional_occupancy(relative_direction, durations_temp, n_bins=12)

            # add the directional occupancy to positional_occupancy_temp
            reldir_occ_by_pos[s, p, :] = directional_occupancy

    return reldir_occ_by_pos


def get_directional_occupancy(directions, durations, n_bins=12):

    # create 24 bins, each 15 degrees
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

def get_directions_to_position(point_in_space, positions):
    """ Suppose point in space (sink) is at (50,50) and position (animal) is (0,0)
    Then the sink is bottom right from the animal (since y increases downwards in image coordinates)
    Therefore the angle is -45 degrees or -pi/4 radians. This checks out
    x_diff = 50 - 0 = 50
    y_diff = 0 - 50 = -50
    direction = arctan(-50/50) = arctan(-1) = -pi/4
    """
    x_diff = point_in_space[0] - positions['x']
    y_diff = positions['y'] - point_in_space[1]
    directions = np.arctan2(y_diff, x_diff)
    return directions


def get_relative_directions_to_position(directions_to_position, head_directions):
    
    relative_direction = head_directions - directions_to_position
    # any relative direction greater than pi is actually less than pi
    relative_direction[relative_direction > np.pi] -= 2*np.pi
    # any relative direction less than -pi is actually greater than -pi
    relative_direction[relative_direction < -np.pi] += 2*np.pi
    
    return relative_direction

