
import numpy as np
import pandas as pd
cm_per_pixel = 0.2


def get_directional_occupancy_by_position(dlc_data, limits):

    # NOTE THAT IN THE OUTPUT, THE FIRST INDEX IS THE Y AXIS, 
    # AND THE SECOND INDEX IS THE X AXIS

    x_bins, y_bins = get_xy_bins(limits, n_bins=100)

    x_bins_og = get_og_bins(x_bins)
    y_bins_og = get_og_bins(y_bins)

    # create positional occupancy matrix
    n_x_bins = len(x_bins) - 1
    n_y_bins = len(y_bins) - 1

    # create directional occupancy by position array
    n_dir_bins=12 # 12 bins of 30 degrees each
    directional_occupancy_temp = np.zeros((n_y_bins, n_x_bins, n_dir_bins))

    # get x and y data 
    x = dlc_data['x']
    y = dlc_data['y']

    x_bin = np.digitize(x, x_bins) - 1
    x_bin[x_bin == n_x_bins] = n_x_bins - 1
    y_bin = np.digitize(y, y_bins) - 1 
    y_bin[y_bin == n_y_bins] = n_y_bins - 1

    # get the head direction
    hd = dlc_data['hd']

    # get the durations
    durations = dlc_data['durations']

    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            # get the head directions and durations for these indices
            hd_temp = hd[indices]
            durations_temp = durations[indices]

            # get the directional occupancy for these indices
            directional_occupancy, direction_bins = \
                get_directional_occupancy(hd_temp, durations_temp, n_bins=n_dir_bins)

            # add the directional occupancy to positional_occupancy_temp
            directional_occupancy_temp[j, i, :] = directional_occupancy

    directional_occupancy_temp = np.round(directional_occupancy_temp, 3)

    directional_occupancy_by_position = {'occupancy': directional_occupancy_temp, 
                            'x_bins': x_bins_og, 'y_bins': y_bins_og, 'direction_bins': direction_bins}

    return directional_occupancy_by_position



def get_relative_direction_occupancy_by_position(pos_data, limits, n_dir_bins = 12, frame_rate = 25):
    '''
    output is a y, x, y, x, n_bins array.
    The first y and x are the position bins, and the second y and x are the consink positions.     
    '''
        
    # get spatial bins
    x_bins, y_bins = get_xy_bins(limits, n_bins=120)

    # candidate consink positions will be at bin centres
    # get_og_bins just adjusts the last bin a bit, doesn't really do anything for us
    x_bins_og = get_og_bins(x_bins)
    y_bins_og = get_og_bins(y_bins)

    bins = {'x': x_bins_og, 'y': y_bins_og}

    # sink position in centres
    x_sink_pos = x_bins_og[0:-1] + np.diff(x_bins_og)/2
    y_sink_pos = y_bins_og[0:-1] + np.diff(y_bins_og)/2

    candidate_sinks = {'x': x_sink_pos, 'y': y_sink_pos}

    # create positional occupancy matrix
    n_x_bins = len(x_bins) - 1
    n_y_bins = len(y_bins) - 1

    # create relative directional occupancy by position array
    
    reldir_occ_by_pos = np.array([[np.zeros((n_y_bins, n_x_bins, n_dir_bins)) for _ in range(n_x_bins)] for _ in range(n_y_bins)])
    
    # get x and y data
    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()
    
    # Remove nan values, otherwise binning gets funky
    mask = np.isnan(hd)|np.isnan(x)
    x = x[~mask]
    y = y[~mask]
    hd = hd[~mask]

    # Binning data and making it 0 indexing based
    x_bin = np.digitize(x, x_bins) - 1 
    # find x_bin == n_x_bins, and set it to n_x_bins - 1
    x_bin[x_bin == n_x_bins] = n_x_bins - 1 

    y_bin = np.digitize(y, y_bins) - 1
    y_bin[y_bin == n_y_bins] = n_y_bins - 1


    # Going over positional bins
    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]
            try:
                x_positions = x[indices]
                y_positions = y[indices]
            except:
                print(i, j)
                breakpoint()
            positions = {'x': x_positions, 'y': y_positions}

            # get the head directions and durations for these indices
            hd_temp = hd[indices]


            # loop through possible consink positions
            for i2, x_sink in enumerate(x_sink_pos):
                for j2, y_sink in enumerate(y_sink_pos):
                
                    # get directions to sink  
                    # NOTE: direction function works well (Sophia)                  
                    directions = get_directions_to_position([x_sink, y_sink], positions)
                    
                    # get the relative direction
                    relative_direction = get_relative_directions_to_position(directions, hd_temp)
                    durations_temp = np.ones(len(relative_direction))/frame_rate #NOTE: Jake's code used durations data from DLC.
                    #We don't use that, each frame is equally long, so we replace all durations with 1/frame_rate

                    # get the directional occupancy for these indices
                    directional_occupancy, direction_bins = \
                        get_directional_occupancy(relative_direction, durations_temp, n_bins=12)

                    # add the directional occupancy to positional_occupancy_temp
                    reldir_occ_by_pos[j, i, j2, i2, :] = directional_occupancy

    return reldir_occ_by_pos, bins, candidate_sinks


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


def get_positional_occupancy(dlc_data, limits):    

    # NOTE THAT IN THE OUTPUT, THE FIRST INDEX IS THE Y AXIS, 
    # AND THE SECOND INDEX IS THE X AXIS

    x_bins, y_bins = get_xy_bins(limits, n_bins=400)

    x_bins_og = get_og_bins(x_bins)
    y_bins_og = get_og_bins(y_bins)

    # create positional occupancy matrix
    n_x_bins = len(x_bins) - 1
    n_y_bins = len(y_bins) - 1
    positional_occupancy_temp = np.zeros((n_y_bins, n_x_bins))

    # get x and y data 
    x = dlc_data['x']
    y = dlc_data['y']

    x_bin = np.digitize(x, x_bins) - 1
    x_bin[x_bin == n_x_bins] = n_x_bins - 1
    y_bin = np.digitize(y, y_bins) - 1 
    y_bin[y_bin == n_y_bins] = n_y_bins - 1

    # sort the x and y bins into the positional occupancy matrix
    for i, (x_ind, y_ind) in enumerate(zip(x_bin, y_bin)):        
        positional_occupancy_temp[y_ind, x_ind] += dlc_data['durations'][i]
        
    positional_occupancy_temp = np.round(positional_occupancy_temp, 3)

    positional_occupancy = {'occupancy': positional_occupancy_temp, 
                            'x_bins': x_bins_og, 'y_bins': y_bins_og}

    return positional_occupancy


def get_og_bins(bins):
   
    # get differences between each bin
    bin_diffs = np.diff(bins)

    # the last difference should be greater than the others, determine by how much
    diff = bin_diffs[-1] - np.mean(bin_diffs[:-1])

    # substract this difference from the last bin
    bins[-1] = bins[-1] - diff

    return bins


def calculate_frame_durations(dlc_data):

    def calculate_intervals(times):
        frame_intervals = np.diff(times)
        # one less interval than frames, so we'll just replicate the last interval
        frame_intervals = np.append(frame_intervals, frame_intervals[-1])

        # add frame intervals to dlc_data
        frame_intervals = frame_intervals/1000

        # round to 4 decimal places, i.e. 0.1 ms
        frame_intervals = np.round(frame_intervals, 4)

        return frame_intervals

    for t, d in dlc_data.items():

        # if d is list then loop through the list, otherwise calculate frame intervals from d
        if isinstance(d, list):
            for i, d2 in enumerate(d):
                times = d2['ts'].values
                
                frame_intervals = calculate_intervals(times)

                dlc_data[t][i]['durations'] = frame_intervals
        
        else:
            times = dlc_data[d]['ts'].values
            
            frame_intervals = calculate_intervals(times)
            
            dlc_data[d]['durations'] = frame_intervals

    return dlc_data


def get_axes_limits(dlc_data = None, x_min = 550, x_max = 2050, y_min = 350, y_max = 1850):
    # CHECK WHETHER X AND Y ARE CORRECT
    # get the x and y limits of the maze
    # By default, we're using limits that we define ourselves, not limits from the data
    if dlc_data is None:
        x_width = x_max - x_min
        y_height = y_max - y_min
    else:
        x_min = np.min(dlc_data['x'])
        x_max = np.max(dlc_data['x'])
        x_width = x_max - x_min

        y_min = np.min(dlc_data['y'])
        y_max = np.max(dlc_data['y'])    
        y_height = y_max - y_min

    limits =  {'x_min': x_min, 'x_max': x_max, 'x_width': x_width,
            'y_min': y_min, 'y_max': y_max, 'y_height': y_height}

    return limits


def get_consink_candidate_positions(dlc_data, limits, n_xbins=None, n_ybins=None):
    # get the x and y limits of the maze
    x_min = limits['x_min']
    x_max = limits['x_max']

    y_min = limits['y_min']
    y_max = limits['y_max']

    if n_xbins is None:
        n_xbins = 10
    if n_ybins is None:
        n_ybins = 10

    # create bins
    x_bins = np.linspace(x_min, x_max, n_xbins + 1)
    y_bins = np.linspace(y_min, y_max, n_ybins + 1)

    return x_bins, y_bins


def get_direction_bins(n_bins=12):
    direction_bins = np.linspace(-np.pi, np.pi, n_bins+1)
    return direction_bins


def bin_directions(directions, direction_bins):
    # get the bin indices for each value in directions
    bin_indices = np.digitize(directions, direction_bins, right=True) - 1
    # any bin_indices that are -1 should be 0
    bin_indices[bin_indices==-1] = 0
    # any bin_indices that are n_bins should be n_bins-1
    n_bins = len(direction_bins) - 1
    bin_indices[bin_indices==n_bins] = n_bins-1

    # get the counts for each bin
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        counts[i] = np.sum(bin_indices==i)

    return counts, bin_indices


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

