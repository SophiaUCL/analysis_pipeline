import numpy as np
import os
import pandas as pd
import json
from typing import Literal
import astropy.convolution as cnv
from pathlib import Path
from spikeinterface.core import BaseRecording, BaseSorting, SortingAnalyzer
UnitTypes = Literal['pyramidal', 'good', 'all']
Methods = Literal["ears", "center"]

def load_unit_ids(derivatives_base: Path, unit_type: UnitTypes, unit_ids: list) -> list:
    """ Returns unit_ids, the unit_ids that we will create rmaps for"""
    if unit_type == 'good':
        good_units_path =derivatives_base/"ephys"/"concat_run"/"sorting"/"sorter_output"/"cluster_group.tsv"
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
    elif unit_type == 'pyramidal':
        pyramidal_units_path = derivatives_base/"analysis"/"cell_characteristics"/"unit_features"/"all_units_overview"/"pyramidal_units_2D.csv"
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
        print("Using pyramidal units")
    elif unit_type == "all":
        print("Using all units")
        unit_ids = unit_ids
    else:
        raise ValueError("unit_type not good, pyramidal, or all. Provide correct input")
    return unit_ids
 
def get_limits(derivatives_base: Path) -> tuple[float, float, float, float]:
    """ Reads in limits from limits.json"""
    limits_path = derivatives_base/"analysis"/"maze_overlay"/"limits.json"
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    
    xmin = limits['x_min']
    xmax = limits['x_max']
    ymin = limits['y_min']
    ymax = limits['y_max']
    return xmin, xmax, ymin, ymax

def get_outline(derivatives_base: Path):
    """Obtains outline of maze from maze_outline_coords.json"""
    outline_path = derivatives_base/"analysis"/"maze_overlay"/"maze_outline_coords.json"
    if outline_path.exists():
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        print("Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None     
    return outline_x, outline_y

def get_trial_length_info(epoch_times, trials_length,  tr):
    """ Returns start time of trial and trial length"""
    trial_row = epoch_times[(epoch_times.trialnumber == tr)]
    start_time = trial_row.iloc[0, 1]

    trial_length_row = trials_length[(trials_length.trialnumber == tr)]
    trial_length = trial_length_row.iloc[0, 2]
    return start_time, trial_length, trial_row
            
def get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate = 25):
    """ Restricts spiketrain to current trial
    Expects input in frames"""
    spike_train_this_trial = np.copy(spike_train)
    spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)*frame_rate] # filtering for current trial
    spike_train_this_trial = [el - np.round(trial_dur_so_far*frame_rate) for el in spike_train_this_trial]
    spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)]
    return spike_train_this_trial

def get_spikes_epoch(spike_train_this_trial, epoch_start, epoch_end, frame_rate):
    """ Restricts spiketrain to this epoch"""
    spike_train_this_epoch = [np.int32(el) for el in spike_train_this_trial if el > frame_rate*epoch_start and el < frame_rate *epoch_end]
    spike_train_this_epoch = np.asarray(spike_train_this_epoch, dtype=int)
    return spike_train_this_epoch
                     
def get_posdata(derivatives_base: Path, method = "ears", g = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """ Loads pos data path. g corresponds to the goal, where g == 3 is the full trial, g == 4 is open field trial (if the last trial is open field)"""
    if g is None or g == 3:
        if method == "ears":
            pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
        elif method == "center":
            pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv')
        else:
            raise ValueError("Method must be ears or center")
    elif g in [0,1,2]:
         pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{g}_trials.csv')
    elif g == 4:
        config_path = os.path.join(derivatives_base, "config.json")
        with open(config_path) as json_data:
            configs = json.load(json_data)
            json_data.close()
        inputs = configs["inputs"]
        trial_numbers = inputs["trial_numbers"]
        open_field_trial = trial_numbers[-1] # assumes last trial is open field
        pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_center_t{open_field_trial}.csv')
    pos_data = pd.read_csv(pos_data_path)

    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()

    return x, y, hd, pos_data

def get_occupancy_time(hd: np.ndarray, frame_rate: int = 25, num_bins: int = 24) -> np.ndarray:
    """ Obtains occupancy time for each bin for hd"""
    hd_filtered = hd[~np.isnan(hd)]
    if np.nanmax(hd_filtered) > 2*np.pi:
        hd_filtered= np.deg2rad(hd_filtered)
    occupancy_counts, _ = np.histogram(hd_filtered, bins=num_bins, range = [-np.pi, np.pi])
    occupancy_time = occupancy_counts / frame_rate 
    return occupancy_time

def get_spike_train_frames(sorting: BaseSorting, unit_id: int, x = None, sample_rate: int = 30000, frame_rate: int = 25) -> np.ndarray:
    """ Returns spike train in frames. Excludes values above len(x) if x is not none"""
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_pre = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data
    if x is not None:
        spike_train = [np.int32(el) for el in spike_train_pre if el < len(x)]  # Ensure spike train is within bounds of x and y
    else:
        spike_train = [np.int32(el) for el in spike_train_pre]
    return spike_train

def get_unit_info(df_unit_metrics, unit_id):
    """ Loads unit firing rate and label for unit = unit_id"""
    row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]
    unit_firing_rate = row['firing_rate'].values[0]
    unit_label = row['label'].values[0]
    return unit_firing_rate, unit_label

def load_trial_xpos(pos_data_dir, tr):
    """ Returns x pos for trial tr"""
    trial_csv_name = f'XY_HD_t{tr}.csv'
    trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
    xy_hd_trial = pd.read_csv(trial_csv_path)
                
    x = xy_hd_trial.iloc[:, 0].to_numpy()  
    return x 

def get_directional_firingrate(hd: np.ndarray, spike_train: np.ndarray | list, num_bins: int, occupancy_time: np.ndarray):
    """ Gets the diretional firing rate and the bin centers"""
    
    # Get counts per bin
    spikes_hd = hd[spike_train]
    spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
    if np.nanmax(hd) > 2*np.pi:
        spikes_hd_rad = np.deg2rad(spikes_hd)
    else:
        spikes_hd_rad = spikes_hd
    counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )
    # Calculating directional firing rate
    direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)

    # Getting bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return direction_firing_rate, bin_centers
        

def get_goal_numbers(derivatives_base: Path):
    """
    Obtains goal numbers from alltrials_trialday.csv
    
    Returns:
        [goal1, goal2]
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    df_path =rawsession_folder/"behaviour"/"alltrials_trialday.csv"
    df = pd.read_csv(df_path)
    goal1 = df['Goal 1'].values[0]
    goal2 = df['Goal 2'].values[0]
    
    ## Gets goal coordinates
    params_path = derivatives_base/"analysis"/"maze_overlay"/"maze_overlay_params.json"
    with open(params_path) as f:
        params= json.load(f)
    hcoord_tr = params["hcoord_tr"]
    vcoord_tr= params["vcoord_tr"]
    
    goal1_coords = [hcoord_tr[np.int32(goal1 -1)], vcoord_tr[np.int32(goal1 - 1)]]
    goal2_coords = [hcoord_tr[np.int32(goal2 -1)], vcoord_tr[np.int32(goal2 - 1)]]
    
    coords_path =  derivatives_base/"analysis"/"maze_overlay"/"goal_coords.json"
    coords = {
        "goal1_coords": goal1_coords,
        "goal2_coords": goal2_coords
    }
    coords_path.parent.mkdir(exist_ok = True, parents = True)
    with open(coords_path, 'w') as f:
        json.dump(coords, f, indent=4)
        
    return [np.int32(goal1), np.int32(goal2)]

def add_relative_hd(derivatives_base, goal_coordinates, method = "ears", goals = [1,2, 3]):
    """ Adds relative hd as a column to the positional data files for each goal
    3 corresponds to full trial
    method corresponds to whether we look at ears or center"""
    
    for i, goal in enumerate(goals):
        # Load positional data
        if goal == 1 or goal == 2:
            pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{goal}_trials.csv')
            pos_data = pd.read_csv(pos_data_path)
            x = pos_data.iloc[:, 0].to_numpy()
            y = pos_data.iloc[:, 1].to_numpy()
            hd = pos_data.iloc[:, 2].to_numpy()
            
            if np.nanmax(hd) > 2*np.pi:
                hd = np.deg2rad(hd)
            positions = {'x': x, 'y': y}
            goal_loc = goal_coordinates[i]
            goaldir_allframes = np.zeros((len(x),1))
            directions = get_directions_to_position([goal_loc[0], goal_loc[1]], positions)
            goaldir_allframes[:, 0] = directions
            relative_direction = get_relative_directions_to_position(directions, hd)
            pos_data['relative_hd'] = relative_direction
            pos_data.to_csv(pos_data_path, index=False)
        elif goal == 3:
            if method == "ears":
                fulltrial_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
            elif method == "center":
                fulltrial_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv')
            goals_without_g3 = [g for g in goals if g !=3]
            
            pos_data = pd.read_csv(fulltrial_path)
            x = pos_data.iloc[:, 0].to_numpy()
            y = pos_data.iloc[:, 1].to_numpy()
            hd = pos_data.iloc[:, 2].to_numpy()
            if np.nanmax(hd) > 2*np.pi:
                hd = np.deg2rad(hd)
            for j, g in enumerate(goals_without_g3):
                positions = {'x': x, 'y': y}
                goal_loc = goal_coordinates[j]
                goaldir_allframes = np.zeros((len(x),1))
                directions = get_directions_to_position([goal_loc[0], goal_loc[1]], positions)
                goaldir_allframes[:, 0] = directions
                relative_direction = get_relative_directions_to_position(directions, hd)
                pos_data[f'relative_hd_g{g}'] = relative_direction
                pos_data.to_csv(fulltrial_path, index=False)

def get_ratemaps(spikes: np.ndarray, x: np.ndarray, y: np.ndarray, n: int = 3, binsize: int = 36, stddev: int = 25, occupancy_threshold: float = 0.4, frame_rate: int = 25):
    """
    Calculate the rate map for given spikes and positions.

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int: 15): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.
        occupancy_threshold (float: 0.4): occupancy values below 0.4 get set to nan
        frame_rate (int: 25): frame rate of camera

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_no_nan = x[~np.isnan(x)]
    y_no_nan = y[~np.isnan(y)]
    
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    pos_binned, x_edges, y_edges = np.histogram2d(x_no_nan, y_no_nan, bins=[x_bins, y_bins])
    pos_binned = pos_binned/frame_rate
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_x_no_nan = spikes_x[~np.isnan(spikes_x)]
    spikes_y_no_nan = spikes_y[~np.isnan(spikes_y)]
    spikes_binned, _, _ = np.histogram2d(spikes_x_no_nan, spikes_y_no_nan, bins=[x_bins, y_bins])
    

    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    # Removing values with very low occupancy (these sometimes have very large firing rate)
    rmap[smoothed_pos < occupancy_threshold] = np.nan

    return rmap, x_edges, y_edges


def get_ratemaps_restrictedx(spikes: np.ndarray, x: np.ndarray, y: np.ndarray, x_restr: np.ndarray, y_restr: np.ndarray,  n: int = 3, binsize: int = 36, stddev: int = 25, occupancy_threshold: float = 0.4,frame_rate: int = 25):
    """
    Calculate the rate map for given spikes and positions. x_restr and y_restr are used to calculate the occupancy map (since we're restricting over a time interval here)
    used for plot_rmap_interactive. For example used if you want to look only at one goal 

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_no_nan = x_restr[~np.isnan(x_restr)]
    y_no_nan = y_restr[~np.isnan(y_restr)]
    
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    # Pos: only for restricted data
    pos_binned, x_edges, y_edges = np.histogram2d(x_no_nan, y_no_nan, bins=[x_bins, y_bins])
    pos_binned = pos_binned/frame_rate
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_x_no_nan = spikes_x[~np.isnan(spikes_x)]
    spikes_y_no_nan = spikes_y[~np.isnan(spikes_y)]
    spikes_binned, _, _ = np.histogram2d(spikes_x_no_nan, spikes_y_no_nan, bins=[x_bins, y_bins])
    

    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    # Removing values with very low occupancy (these sometimes have very large firing rate)
    rmap[smoothed_pos < occupancy_threshold] = np.nan
    return rmap, x_edges, y_edges

            

def get_relative_directions_to_position(directions_to_position, head_directions):
    
    relative_direction = head_directions - directions_to_position
    # any relative direction greater than pi is actually less than pi
    relative_direction[relative_direction > np.pi] -= 2*np.pi
    # any relative direction less than -pi is actually greater than -pi
    relative_direction[relative_direction < -np.pi] += 2*np.pi
    
    return relative_direction

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
        
        
    
def get_goal_coordinates(derivatives_base: Path, rawsession_folder: Path):
    """
    Returns:
        Goal coordinates. If json file with them doesn't exist, it makes it
    """
    coords_path =  derivatives_base/"analysis"/"maze_overlay"/"goal_coords.json"
    if not coords_path.exists():
        get_goal_numbers(derivatives_base)
    
    with open(coords_path) as f:
        data= json.load(f)

    goal1_coords = data["goal1_coords"]
    goal2_coords = data["goal2_coords"]
    return [goal1_coords, goal2_coords]


