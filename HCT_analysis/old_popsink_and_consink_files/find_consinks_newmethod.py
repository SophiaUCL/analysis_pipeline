import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from calculate_pos_and_dir import  get_directions_to_position, get_relative_directions_to_position
from calculate_occupancy import get_relative_direction_occupancy_by_position,  get_axes_limits,  get_direction_bins, bin_directions
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from utilities.trials_utils import get_limits_from_json, get_goal_coordinates, get_coords
from matplotlib.patches import RegularPolygon
import matplotlib
matplotlib.use("QtAgg") 
from joblib import Parallel, delayed
from utilities.mrl_func import resultant_vector_length
from astropy.stats import circmean
from tqdm import tqdm
from typing import Literal
cm_per_pixel = 1
import warnings

def rel_dir_distribution_all_sinks(spike_train, sink_bins, candidate_sinks, direction_bins, pos_data,  reldir_allframes):
   
    """ 
    Create array to store the relative direcion histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 3D array, with
    dimensions (n_y_bins, n_x_bins, n_direction_bins). 

    PRETTY SURE THIS DOESN'T NEED THE FIRST SET OF X AND Y LOOPS!!!!!!!!!!!!!!! FIX IT!!!!!!!!!! <- note by Jake, unsure if true

    """
    
    # create array to store relative direction histograms
    rel_dir_dist = np.zeros((len(candidate_sinks['y']), len(candidate_sinks['x']), len(direction_bins) - 1))  


    # loop through candidate consink positions
    for i2, x_sink in enumerate(candidate_sinks['x']):
        for j2, y_sink in enumerate(candidate_sinks['y']):
            reldirections_sink = reldir_allframes[:, j2, i2]

            # get the relative direction
            relative_direction = reldirections_sink[spike_train]

            relative_direction = relative_direction[~np.isnan(relative_direction)]
            # bin the relative directions
            rel_dir_binned_counts, _ = bin_directions(relative_direction, direction_bins)
            rel_dir_dist[j2, i2, :] = rel_dir_dist[j2, i2, :] + rel_dir_binned_counts
    if np.all(rel_dir_dist == 0):
        print("rel dir dist all 0")
    return rel_dir_dist

def get_dir_allframes(pos_data, candidate_sinks):
    """ Gets directions from each frame to each sink"""

    sinkdir_allframes = np.zeros(
        (len(pos_data), len(candidate_sinks['y']), len(candidate_sinks['x']))
    )

    reldir_allframes = np.zeros(
        (len(pos_data), len(candidate_sinks['y']), len(candidate_sinks['x']))
    )


    x_org = pos_data.iloc[:, 0].to_numpy()
    y_org = pos_data.iloc[:, 1].to_numpy()
    hd_org = pos_data.iloc[:, 2].to_numpy()
    positions = {'x': x_org, 'y': y_org}
    for i2, x_sink in enumerate(candidate_sinks['x']):
        for j2, y_sink in enumerate(candidate_sinks['y']):
            directions = get_directions_to_position([x_sink, y_sink], positions)
            sinkdir_allframes[:, j2, i2] = directions
            relative_direction = get_relative_directions_to_position(directions, hd_org)
            reldir_allframes[:, j2, i2] =relative_direction
    return sinkdir_allframes, reldir_allframes

def rel_dir_ctrl_distribution_all_sinks(spike_train, reldir_occ_by_pos, sink_bins, candidate_sinks, pos_data):
    """
    For a given unit, produces relative direction occupancy distributions 
    for each candidate consink position based on the number of spikes fired 
    at each positional bin. 
    
    NOTE: The input for the spikes will already be restricted to each goal
    """

    # get head directions as np array
    x_bin_all = pos_data['x_bin'].to_numpy()
    y_bin_all = pos_data['y_bin'].to_numpy()
    x_bin = x_bin_all[spike_train]
    y_bin = y_bin_all[spike_train]

    mask = np.isnan(x_bin)
    x_bin = x_bin[~mask]
    y_bin = y_bin[~mask]

    direction_bins = get_direction_bins(n_bins=12)
    rel_dir_ctrl_dist = np.zeros((len(candidate_sinks['y']), len(candidate_sinks['x']), len(direction_bins) -1))

    # loop through the x and y bins
    n_spikes_total = 0

    rows = int(np.max(y_bin)) + 1
    cols = int(np.max(x_bin)) + 1

    indices_all = np.zeros((rows, cols))
    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            indices_all[j,i] = len(indices)
            # Number of spikes this cell fired in the bin
            n_spikes = len(indices)
            if n_spikes == 0:
                continue
            # number of spikes cell fired in total so far
            n_spikes_total = n_spikes_total + n_spikes

            # Add (n_y_sinks, n_x_sinks, n_dir_bins)*n_spikes (scale by how many spikes are fired there)
            rel_dir_ctrl_dist = rel_dir_ctrl_dist + reldir_occ_by_pos[j,i,:,:,:] * n_spikes

    if np.all(rel_dir_ctrl_dist == 0):
        #print("All zeroes in rel dir ctrl dist. Breakpoint")
        pass
    return rel_dir_ctrl_dist, n_spikes_total

def normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total):
    """
    Normalise the relative direction distribution by the control distribution. 
    """

    # first, divide rel_dir_dist by rel_dir_ctrl_dist
    rel_dir_dist_div_ctrl = np.divide(
        rel_dir_dist,
        rel_dir_ctrl_dist,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where=rel_dir_ctrl_dist != 0
    )

    # now we want the counts in each histogram to sum to the total number of spikes
    if len(rel_dir_dist_div_ctrl.shape) > 1:
        sum_rel_dir_dist_div_ctrl = rel_dir_dist_div_ctrl.sum(axis=2)
        sum_rel_dir_dist_div_ctrl_ex = sum_rel_dir_dist_div_ctrl[:,:,np.newaxis]

    else:
        sum_rel_dir_dist_div_ctrl_ex = rel_dir_dist_div_ctrl.sum()

    normalised_rel_dir_dist = np.divide(
        rel_dir_dist_div_ctrl,
        sum_rel_dir_dist_div_ctrl_ex,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where=sum_rel_dir_dist_div_ctrl_ex != 0
    )
    normalised_rel_dir_dist = normalised_rel_dir_dist* n_spikes_total

    return normalised_rel_dir_dist


def mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins):
    """
    Calculate the mean resultant length of the normalised relative direction distribution. 
    """

    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1])/2

    n_y_bins = normalised_rel_dir_dist.shape[0]
    n_x_bins = normalised_rel_dir_dist.shape[1]

    mrl = np.zeros((n_y_bins, n_x_bins))
    mean_angle = np.zeros((n_y_bins, n_x_bins))

    for i in range(n_y_bins):
        for j in range(n_x_bins):

            mrl[i,j] = resultant_vector_length(dir_bin_centres, w=normalised_rel_dir_dist[i,j,:])

            mean_angle[i,j] = circmean(dir_bin_centres, weights=normalised_rel_dir_dist[i,j,:])

            #warnings.filterwarnings("error")
    return mrl, mean_angle

def plot_all_consinks(consinks_df, goal_coordinates, hcoord, vcoord,  limits, jitter, plot_dir, plot_name='ConSinks'):
    num_goals = 3
    # Create two subplots
    fig, axes = plt.subplots(1,num_goals, figsize = (30,10))
    axes = axes.flatten()
    fig.suptitle(plot_name, fontsize=24)

    for g in range(num_goals):
        ax = axes[g]
        ax.set_xlabel('x position (cm)', fontsize=16)
        ax.set_ylabel('y position (cm)', fontsize=16)

        # Add some coloured hexagons and adjust the orientation to match the rotated grid
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            hex = RegularPolygon((x, y), numVertices=6, radius=83,
                                orientation=np.radians(28),  # Rotate hexagons to align with grid
                                facecolor='grey', alpha=0.2, edgecolor='k')
            ax.text(x, y, i + 1, ha='center', va='center', size=15)  # Start numbering from 1
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
        # plot the goal positions
        if g > 0:
            circle = plt.Circle(goal_coordinates[g-1], 80, color='g', 
                    fill=False, linewidth=5)
            ax.add_artist(circle)   
            title = f'Goal {g}'
        else:
            title = 'G1 but going to G2'
        ax.set_title(title, fontsize=20)
        # loop through the rows of the consinks_df, plot a filled red circle at the consink 
        # position if the mrl is greater than ci_999

        clusters = []
        consink_positions = []

        for cluster in consinks_df.index:
            
            x_jitter = np.random.uniform(-jitter[0], jitter[0])
            y_jitter = np.random.uniform(-jitter[1], jitter[1])

            consink_position = consinks_df.loc[cluster, 'position_g'+str(g)]

            sig = consinks_df.loc[cluster, 'sig_g'+str(g)]
            if sig == "sig":
                
                circle = plt.Circle((consink_position[0] + x_jitter, 
                    consink_position[1] + y_jitter), 60, color='r', 
                    fill=True)
                ax.add_artist(circle)  

                clusters.append(cluster)
                consink_positions.append(consink_position)
            
            # set the x and y limits
            ax.set_xlim((limits['x_min']-200, limits['x_max']+200))
            ax.set_ylim(limits['y_min']-200, limits['y_max']+200)

            # reverse the y axis
            ax.invert_yaxis()

            # make the axes equal
            ax.set_aspect('equal')
            """
            # set font size of axes
            ax.tick_params(axis='both', which='major', labelsize=14)

            # get the axes values
            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()

            # convert the axes values to cm
            x_ticks_cm = x_ticks * cm_per_pixel
            y_ticks_cm = y_ticks * cm_per_pixel

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks_cm)

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks_cm)
            """


    plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
    print(f"Saved figure to {os.path.join(plot_dir, plot_name + '.png')}")
    plt.show()

def verify_allnans(spike_train, pos_data):
    """ Verifies that not all values are nan values"""
    x_org = pos_data.iloc[:, 0].to_numpy()
    hd_org = pos_data.iloc[:, 2].to_numpy()

    #spike_train = [el for el in spike_train if el < len(x_org)]  # Ensure spike train is within bounds of x and y
    # Finding spike times for this unit
    x = x_org[spike_train]
    hd = hd_org[spike_train]

    mask = np.isnan(x) | np.isnan(hd)
    false_vals = np.where(mask == False)[0]
    if len(false_vals) < 2:
        return True
    else:
        return False


def find_consink(spike_train, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks, pos_data,  reldir_allframes):
    """
    Find the consink position that maximises the mean resultant length of the normalised relative direction distribution. 
    """
    spike_train = np.array(spike_train)
    all_nans = verify_allnans(spike_train, pos_data)

    if all_nans:
        print("All nans. Returning nan")
        return np.nan, np.nan, np.nan
    #  get control occupancy distribution
    rel_dir_ctrl_dist, n_spikes_total = rel_dir_ctrl_distribution_all_sinks(spike_train, reldir_occ_by_pos, sink_bins, candidate_sinks, pos_data)

    # rel dir distribution for each possible consink position
    rel_dir_dist = rel_dir_distribution_all_sinks(spike_train, sink_bins, candidate_sinks, direction_bins, pos_data,  reldir_allframes)

    # normalise rel_dir_dist by rel_dir_ctrl_dist
    normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total)
    if np.isnan(normalised_rel_dir_dist).any():
        breakpoint()
    # calculate the mean resultant length of the normalised relative direction distribution
    mrl, mean_angle = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)

    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angle[max_mrl_indices[0][0], max_mrl_indices[1][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle


def shift_spiketrain(spike_train, n_frames: int, frame_rate = 25):
    """Shift the spike train by a random amount.

    Args:
        spike_train (array): firing times of unit (in frames)
        n_frames (int): length of the recording (in frames)
    """
    spike_train = np.array(spike_train)
    min_shift = 5 * frame_rate
    max_shift = n_frames - min_shift + 1
    # pick a shift randomly between those two numbers
    shift = np.random.randint(min_shift, max_shift)
    
    shifted_data = spike_train + shift

    range_min = 0
    range_max = n_frames - 1
    range_size = range_max - range_min + 1
    
    # Ensure shifted_data stays within the range [range_min, range_max]
    shifted_data = np.mod(shifted_data - range_min, range_size) + range_min
    
    return shifted_data

def shift_spiketrain_pergoal(spike_train, goal,  intervals_frames, n_frames:int,  frame_rate = 25):
    """Shift the spike train by a random amount. Restrict it to goal intervals

    Args:
        spike_train (array): firing times of unit (in frames)
        goal(int): goal number
        n_frames (int): length of the recording (in frames)
    """
    start_col = goal * 2
    spike_train = np.array(spike_train)
    min_shift = 2 * frame_rate
    lengths = [intervals_frames.iloc[tr, start_col + 1] - intervals_frames.iloc[tr, start_col] for tr in range(len(intervals_frames))]
    max_shift = np.max(lengths) - min_shift + 1


    # pick a shift randomly between those two numbers
    shift = np.random.randint(min_shift, max_shift)

    shifted_data = np.empty(0, dtype=int)


    for tr in range(len(intervals_frames)):
        start_frame = intervals_frames.iloc[tr, start_col]
        end_frame = intervals_frames.iloc[tr, start_col + 1]
        spike_train_tr = spike_train[(spike_train >= start_frame) & (spike_train <= end_frame)]

        if len(spike_train_tr) == 0:
            continue
        shifted_data_tr= spike_train_tr + shift

        range_min = start_frame
        range_max = end_frame
        range_size = range_max - range_min + 1


        shifted_data_tr = np.mod(shifted_data_tr - range_min, range_size) + range_min

        if np.min(shifted_data_tr) < range_min or np.max(shifted_data_tr) > range_max:
            breakpoint()

        shifted_data = np.append(shifted_data, shifted_data_tr)

    shifted_data = np.array(shifted_data)
    return shifted_data

def shift_spiketrain_pergoal_old(spike_train, goal, rawsession_folder, n_frames:int, intervals_frames, frame_rate = 25):
    """Shift the spike train by a random amount. Restrict it to goal intervals

    Args:
        spike_train (array): firing times of unit (in frames)
        goal(int): goal number
        n_frames (int): length of the recording (in frames)
    """
    spike_train = np.array(spike_train)
    min_shift = 2 * frame_rate
    max_shift = n_frames - min_shift + 1

    # pick a shift randomly between those two numbers
    shift = np.random.randint(min_shift, max_shift)

    shifted_data = spike_train + shift

    range_min = 0
    range_max = n_frames - 1
    range_size = range_max - range_min + 1

    # Ensure shifted_data stays within the range [range_min, range_max]
    shifted_data = np.mod(shifted_data - range_min, range_size) + range_min
    shifted_data_secs = shifted_data/25
    restrict_spiketrain_specialbehav(shifted_data_secs, rawsession_folder, goal=goal)

    return shifted_data

def calculate_translated_mrl(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks,  reldir_allframes, rawsession_folder, intervals_frames, goal):
    n_frames = len(dlc_data)
    translated_spiketrain = shift_spiketrain_pergoal(spiketrain, goal, intervals_frames, n_frames)
    mrl, _, _ = find_consink(translated_spiketrain, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks, dlc_data,  reldir_allframes)
    return mrl

def recalculate_consink_to_all_candidates_from_translation(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks,  reldir_allframes, rawsession_folder, intervals_frames, goal):

    n_shuffles = 1000
    mrl = np.zeros(n_shuffles)


    #mrl = Parallel(n_jobs=-1, verbose=0)(delayed(calculate_translated_mrl)(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles))
    mrl = Parallel(n_jobs=-1, verbose=0)(
        delayed(calculate_translated_mrl)(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins,
                                          candidate_sinks,reldir_allframes,  rawsession_folder, intervals_frames, goal) for s in range(n_shuffles))
    #mrl = [calculate_translated_mrl(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles)]
    # remove nan values from mrl
    mrl = np.array(mrl)
    mrl = mrl[~np.isnan(mrl)]
    if len(mrl) == 0:
        return (np.nan, np.nan, np.nan)
    mrl = np.round(mrl, 3)
    mrl_95 = np.percentile(mrl, 95)
    mrl_999 = np.percentile(mrl, 99.9)

    if len(mrl) < 1000:
        print(len(mrl))
    ci = (mrl_95, mrl_999, n_shuffles - len(mrl)) # last one is the length after nans are removed
    
    return ci

def add_bins_posdata(pos_data, sink_bins):
    """ Adds bins to the posdata as an extra column"""

    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    x_bin = np.digitize(x, sink_bins['x']) - 1
    # find x_bin == n_x_bins, and set it to n_x_bins - 1
    if len(x_bin) > 1:
        x_bin[x_bin == (len(sink_bins['x']) - 1)] = len(sink_bins['x']) - 2

    y_bin = np.digitize(y, sink_bins['y']) - 1
    # find y_bin == n_y_bins, and set it to n_y_bins - 1
    if len(y_bin) > 1:
        y_bin[y_bin == (len(sink_bins['y']) - 1)] = len(sink_bins['y']) - 2

    pos_data['x_bin'] = x_bin
    pos_data['y_bin'] = y_bin
    return pos_data

def get_spike_train(sorting, unit_id, pos_data,  rawsession_folder, g, frame_rate = 25, sample_rate = 30000):
    """ Obtains the spike train and restricts it to the goal"""
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_secs = spike_train_unscaled / sample_rate  # This is in seconds now
    # Restrict spiketrain to goal
    if g == 'all':
        spike_train_secs_g = spike_train_secs
    else:
        spike_train_secs_g = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder, goal=g)
    # Now let spiketrain be in frame_rate
    spike_train = np.round(spike_train_secs_g * frame_rate)
    spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]
    return spike_train

    
def main(derivatives_base, rel_dir_occ: Literal['all trials', 'intervals'], unit_type: Literal['pyramidal', 'good', 'all'], code_to_run = [], frame_rate = 25, sample_rate = 30000):
    """
    Code to find consinks, based on Jake's code


    """
    # Path to rawsession folder
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # Loading limits (currently just the whole camera view)
    x_min, x_max, y_min, y_max = get_limits_from_json(derivatives_base)
    limits = get_axes_limits(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    # Getting unit IDs depending on type
    if unit_type == 'good':
        good_units_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting","sorter_output", "cluster_group.tsv")
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
        # Loading pyramidal units
    elif unit_type == 'pyramidal':
        pyramidal_units_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features","all_units_overview", "pyramidal_units_2D.csv")
        print("Getting pyramidal units 2D")
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
    
    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])

    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_goal1_trials.csv')
    pos_data_g1 = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data_g1['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data_g1['hd'] = np.deg2rad(pos_data_g1['hd'])

    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_goal2_trials.csv')
    pos_data_g2 = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data_g2['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data_g2['hd'] = np.deg2rad(pos_data_g2['hd'])
    
    # Getting the positional data we'll use for the rel_dir_occ
    if rel_dir_occ == 'all trials':
        name = 'XY_HD_alltrials.csv'
    elif rel_dir_occ == 'intervals':
        name = 'XY_HD_allintervals.csv'
    pos_data_reldir_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', name)
    pos_data_reldir = pd.read_csv(pos_data_reldir_path)
   
    if np.nanmax(pos_data_reldir['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data_reldir['hd'] = np.deg2rad(pos_data_reldir['hd']) 
    

    # restricted df frames
    path = os.path.join(rawsession_folder, 'task_metadata', 'restricted_df_frames.csv')
    intervals_frames = pd.read_csv(path)

    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics',  'spatial_features', 'consink_data_newmethod')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Direction bins (from -pi to pi)
    direction_bins = get_direction_bins(n_bins=12)

    # Loading or creating data 
    file_name = 'reldir_occ_by_pos.npy'
    
    if -1 in code_to_run:
        print("Calculating relative direction occupancy by position")
        reldir_occ_by_pos, sink_bins, candidate_sinks = get_relative_direction_occupancy_by_position(pos_data_reldir, limits)
        reldir_occ_by_pos_g1, _, _ = get_relative_direction_occupancy_by_position(pos_data_g1, limits)
        reldir_occ_by_pos_g2, _, _ = get_relative_direction_occupancy_by_position(pos_data_g2, limits)
        np.save(os.path.join(output_folder , file_name), reldir_occ_by_pos)
        np.save(os.path.join(output_folder , 'reldir_occ_by_pos_g1.npy'), reldir_occ_by_pos_g1)
        np.save(os.path.join(output_folder , 'reldir_occ_by_pos_g2.npy'), reldir_occ_by_pos_g2)
        
        # save sink bins and candidate sinks as pickle files
        save_pickle(sink_bins, 'sink_bins', output_folder)
        save_pickle(candidate_sinks, 'candidate_sinks', output_folder)
        save_pickle(direction_bins, 'direction_bins', output_folder)     

    else:
        print("Loading reldir occ, not callculating")
        reldir_occ_by_pos = np.load(os.path.join(output_folder, file_name))
        reldir_occ_by_pos_g1 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy'))
        reldir_occ_by_pos_g2 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g2.npy'))
        sink_bins = load_pickle('sink_bins', output_folder)
        candidate_sinks = load_pickle('candidate_sinks', output_folder)

    ## Get goal coordinates
    # Doesn't do antyhing yet
    goal_coordinates = get_goal_coordinates(derivatives_base, rawsession_folder)
    pos_data = add_bins_posdata(pos_data, sink_bins)
    sinkdir_allframes, reldir_allframes = get_dir_allframes(pos_data, candidate_sinks)

    ################# CALCULATE CONSINKS ###########################################     
    consinks = {}
    consinks_df = {}

    if 0 in code_to_run:
        print("Calculating consinks")
        for unit_id in tqdm(unit_ids):
            consinks[unit_id] = {'unit_id': unit_id}

            for g in [0, 1, 2]:
                if g == 0:
                     # store with goal suffix
                    consinks[unit_id][f'mrl_g{g}'] = np.nan
                    consinks[unit_id][f'position_g{g}'] = np.nan
                    consinks[unit_id][f'mean_angle_g{g}'] = np.nan
                    consinks[unit_id][f'numspikes_g{g}'] = np.nan
                    continue
                if g == 1:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                elif g == 2:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
                else:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos
                    
                # Find spiketrain
                spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
                spike_train_secs = spike_train_unscaled / sample_rate # This is in seconds now
                # Restrict spiketrain to goal
                spike_train_secs_g = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder, goal=g)
                # Now let spiketrain be in frame_rate
                spike_train = np.round(spike_train_secs_g * frame_rate)
                spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]  

                # Skip empty spikes
                if len(spike_train) == 0:
                    continue
                
                # get consink  
                max_mrl, max_mrl_indices, mean_angle = find_consink(
                    spike_train, reldir_occ_by_pos_cur, sink_bins, direction_bins, candidate_sinks, pos_data, reldir_allframes
                )
                consink_position = np.round(
                    [candidate_sinks['x'][max_mrl_indices[1][0]], 
                    candidate_sinks['y'][max_mrl_indices[0][0]]], 3
                )

                # store with goal suffix
                consinks[unit_id][f'mrl_g{g}'] = max_mrl
                consinks[unit_id][f'position_g{g}'] = consink_position
                consinks[unit_id][f'mean_angle_g{g}'] = mean_angle
                consinks[unit_id][f'numspikes_g{g}'] = len(spike_train)

        # Create dataframe
        consinks_df = pd.DataFrame(consinks).T
        print(consinks_df)

        # save as csv            
        consinks_df.to_csv(os.path.join(output_folder, 'consinks_df.csv'), index = False)
        print(f"Data saved to {os.path.join(output_folder, 'consinks_df.csv')}")
        # save consinks_df 
        save_pickle(consinks_df, 'consinks_df', output_folder)

    
    # ######################### TEST STATISTICAL SIGNIFICANCE OF CONSINKS #########################
    # shift the head directions relative to their positions, and recalculate the tuning to the 
    # previously identified consink position. 
    
    if 1 in code_to_run:
        print("Assessing significance")
        # load the consinks_df
        consinks_df = load_pickle('consinks_df', output_folder)

        # make columns for the confidence intervals; place them directly beside the mrl column
        idx_g0 = consinks_df.columns.get_loc('mrl_g0')


        # if the columns don't exist, insert them            
        if 'ci_95_g1' not in consinks_df.columns:
            consinks_df.insert(idx_g0 + 1, 'ci_95_g0', np.nan)
            consinks_df.insert(idx_g0 + 2, 'ci_999_g0', np.nan)
            consinks_df.insert(idx_g0 + 3, 'sig_g0', np.nan)
            idx_g1 = consinks_df.columns.get_loc('mrl_g1')
            consinks_df.insert(idx_g1 + 1, 'ci_95_g1', np.nan)
            consinks_df.insert(idx_g1 + 2, 'ci_999_g1', np.nan)
            consinks_df.insert(idx_g1 + 3, 'sig_g1', np.nan)
            idx_g2 = consinks_df.columns.get_loc('mrl_g2')
            consinks_df.insert(idx_g2 + 1, 'ci_95_g2', np.nan)
            consinks_df.insert(idx_g2 + 2, 'ci_999_g2', np.nan)
            consinks_df.insert(idx_g2 + 3, 'sig_g2', np.nan)

        for unit_id in tqdm(unit_ids):
            for g in [0, 1,2]:
                if g == 0:
                    consinks_df.loc[unit_id, f'ci_95_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'ci_999_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'sig_g{g}'] = np.nan
                    continue
                if g == 1:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                elif g == 2:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
                else:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos
                    
                #print(f'Were at {unit_id} with { g}' )

                if consinks_df.loc[unit_id, f'numspikes_g{g}'] < 30:
                    consinks_df.loc[unit_id, f'ci_95_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'ci_999_g{g}'] = np.nan
                    consinks_df.loc[unit_id, f'sig_g{g}'] = np.nan
                    continue

                spike_train = get_spike_train(sorting, unit_id, pos_data,  rawsession_folder, g = g, frame_rate = frame_rate, sample_rate = sample_rate)


                ci = recalculate_consink_to_all_candidates_from_translation(spike_train, pos_data, reldir_occ_by_pos_cur, sink_bins, direction_bins, candidate_sinks,  reldir_allframes,  rawsession_folder, intervals_frames, goal = g)

                consinks_df.loc[unit_id, f'ci_95_g{g}'] = ci[0]
                consinks_df.loc[unit_id, f'ci_999_g{g}'] = ci[1]
                mrl_val = consinks_df.loc[unit_id, f'mrl_g{g}']
                if np.isfinite(ci[0]) and np.isfinite(mrl_val) and mrl_val > ci[0]:
                    sig = 'sig'
                else:
                    sig = 'ns'
                consinks_df.loc[unit_id, f'sig_g{g}']   = sig

        print(f"Saved consink data to the following folder: {output_folder}")
        try:
            consinks_df.to_csv(os.path.join(output_folder, 'consinks_df.csv'))
        except:
            breakpoint()
        save_pickle(consinks_df, 'consinks_df', output_folder)

    ######################## PLOT ALL CONSINKS #################################
    # calculate a jitter amount to jitter the positions by so they are visible
    x_diff = np.mean(np.diff(candidate_sinks['x']))

    y_diff = np.mean(np.diff(candidate_sinks['y']))
    jitter = (x_diff/3, y_diff/3)
    
    plot_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics',  'spatial_features', 'consink_plots_newmethod')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    hcoord, vcoord = get_coords(derivatives_base)
    # Check if consinks_df is a dictionary otherwise convert
    consinks_df = load_pickle('consinks_df', output_folder)
    

    plot_all_consinks(consinks_df, goal_coordinates, hcoord, vcoord, limits, jitter=jitter, plot_dir=plot_dir, plot_name='ConSinks Good Units')

if __name__ == "__main__":

    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    main(derivatives_base, 'all trials', 'pyramidal', code_to_run=[1])


