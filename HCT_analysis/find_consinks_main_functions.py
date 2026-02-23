import numpy as np
from HCT_analysis.calculate_pos_and_dir import get_directions_to_position, get_relative_directions_to_position
from HCT_analysis.calculate_occupancy import get_direction_bins, bin_directions
from HCT_analysis.utilities.trials_utils import verify_allnans
import matplotlib
matplotlib.use("QtAgg")
from joblib import Parallel, delayed
from HCT_analysis.utilities.mrl_func import resultant_vector_length
from HCT_analysis.utilities.load_and_save_data import load_pickle, save_pickle
from astropy.stats import circmean
import os

num_candidate_sinks = 127
""" In this code, the bins are the platforms"""
import warnings

""" This file contains different functions that we use in find_consinks_main.py to find consinks
Here we implement 3 different methods:
Method 1: normalise the relative direction distribution by the control distribution based on the number of spikes fired at each platform (original method)
Method 2: normalise the relative direction distribution for each platform separately, then sum across platforms
Method 3: normalise the relative direction distribution by the control distribution based on the occupancy for the whole maze (not binned by platform)
"""

def get_dir_allframes(pos_data, sink_positions):
    """ Gets directions from each frame to each sink"""

    sinkdir_allframes = np.zeros(
        (len(pos_data), num_candidate_sinks)
    )

    reldir_allframes = np.zeros(
        (len(pos_data), num_candidate_sinks)
    )


    x_org = pos_data.iloc[:, 0].to_numpy()
    y_org = pos_data.iloc[:, 1].to_numpy()
    hd_org = pos_data.iloc[:, 2].to_numpy()
    positions = {'x': x_org, 'y': y_org}
    
    for s in range(num_candidate_sinks):
        # Position of the sink
        platform_loc = sink_positions[s]
        
        # For each position, calculate the direction to the sink NOTE: what happens to nan values?
        directions = get_directions_to_position([platform_loc[0], platform_loc[1]], positions)
        sinkdir_allframes[:, s] = directions
        
        # For each position, calculate the relative direction to the sink NOTE: what happens to nan values
        relative_direction = get_relative_directions_to_position(directions, hd_org)
        reldir_allframes[:, s] = relative_direction
    return sinkdir_allframes, reldir_allframes

def export_sig_sinks(methods,output_folder,goals_to_include=[0, 1, 2]):
    """
    Export ONLY unit IDs of significant consinks.
    One file per method, containing a dict: {goal: np.array(unit_ids)}
    """

    for method in methods:
        consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)

        sig_units_per_goal = {}

        for g in goals_to_include:
            sig_col = f'sig_g{g}'
            if sig_col not in consinks_df.columns:
                print(f"[Method {method}] No column {sig_col}, skipping")
                continue

            unit_ids = consinks_df.index[
                consinks_df[sig_col] == 'sig'
            ].to_numpy()

            sig_units_per_goal[g] = unit_ids.astype(int)

            print(
                f"[Method {method}, Goal {g}] "
                f"{len(unit_ids)} significant units"
            )

        save_path = os.path.join(
            output_folder,
            f'significant_consink_unit_ids_method_{method}.npy'
        )
        np.save(save_path, sig_units_per_goal, allow_pickle=True)

        print(f"[Method {method}] Saved → {save_path}")

# MRL CALCULATION
def mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins):
    """
    Calculate the mean resultant length of the normalised relative direction distribution.
    """

    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1]) / 2


    mrl = np.zeros(num_candidate_sinks)
    mean_angle = np.zeros(num_candidate_sinks)

    for s in range(num_candidate_sinks):
            mrl[s] = resultant_vector_length(dir_bin_centres, w=normalised_rel_dir_dist[s, :])

            mean_angle[s] = circmean(dir_bin_centres, weights=normalised_rel_dir_dist[s, :])

            # warnings.filterwarnings("error")
    return mrl, mean_angle


# METHOD 1 + 3: SPIKE DISTRIBUTION
def rel_dir_distribution_all_sinks(spike_train, direction_bins, reldir_allframes):
    """
    Create array to store the relative direction histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 2D array, with
    dimensions (n_sinks, n_direction_bins).

    """

    # create array to store relative direction histograms
    rel_dir_dist = np.zeros((num_candidate_sinks, len(direction_bins) - 1))

    # loop through candidate consink positions
    for s in range(num_candidate_sinks):
        reldirections_sink = reldir_allframes[:, s]

        # get the relative direction
        relative_direction = reldirections_sink[spike_train]

        relative_direction = relative_direction[~np.isnan(relative_direction)]
        
        # bin the relative directions
        rel_dir_binned_counts, _ = bin_directions(relative_direction, direction_bins)
        rel_dir_dist[s, :] = rel_dir_dist[s, :] + rel_dir_binned_counts

    if np.all(rel_dir_dist == 0):
        print("rel dir dist all 0")

    return rel_dir_dist

# METHOD 1: CONTROL DISTRIBUTION
def rel_dir_ctrl_distribution_all_sinks(reldir_occ_by_pos, platforms_spk):
    """
    For a given unit, produces relative direction occupancy distributions
    for each candidate consink position based on the number of spikes fired
    at each positional bin.

    NOTE: The input for the spikes will already be restricted to each goal
    """
    direction_bins = get_direction_bins(n_bins=12)
    rel_dir_ctrl_dist = np.zeros((num_candidate_sinks, len(direction_bins) - 1))

    # loop through the x and y bins
    n_spikes_total = 0

    for p in range(61):
        # get the indices where platforms_spj == p + t1
        indices = np.where(platforms_spk == p + 1)[0]

        # Number of spikes this cell fired in the bin
        n_spikes = len(indices)
        if n_spikes == 0:
            continue
        # number of spikes cell fired in total so far
        n_spikes_total = n_spikes_total + n_spikes

        # Add (sink, n_dir_bins)*n_spikes (scale by how many spikes are fired there)
        rel_dir_ctrl_dist = rel_dir_ctrl_dist + reldir_occ_by_pos[:, p,  :] * n_spikes

    if np.all(rel_dir_ctrl_dist == 0):
        # print("All zeroes in rel dir ctrl dist. Breakpoint")
        pass
    return rel_dir_ctrl_dist, n_spikes_total

def get_reldir_bin_idx(reldir_allframes, direction_bins):
    """ For each frame and each candidate sink, get the direction bin index
    Inputs:
    reldir_allframes: np array of shape (num_frames, num_candidate_sinks)
    direction_bins: np array of shape (n_bins + 1,)
    Outputs:
    reldir_bin_idx: np array of shape (num_frames, num_candidate_sinks) with bin indices
    """
    n_bins = len(direction_bins) - 1
    reldir_bin_idx = np.full(reldir_allframes.shape, -1, dtype=np.int8)

    for s in range(num_candidate_sinks):
        valid = ~np.isnan(reldir_allframes[:, s])
        reldir_bin_idx[valid, s] = (
            np.digitize(reldir_allframes[valid, s], direction_bins) - 1
        )
    return reldir_bin_idx


# METHOD 1: NORMALISATION
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
    if len(rel_dir_dist_div_ctrl.shape) > 1: ## NOTE: check whether this is correct here.
        sum_rel_dir_dist_div_ctrl = rel_dir_dist_div_ctrl.sum(axis=1)
        sum_rel_dir_dist_div_ctrl_ex = sum_rel_dir_dist_div_ctrl[:, np.newaxis]

    else:
        sum_rel_dir_dist_div_ctrl_ex = rel_dir_dist_div_ctrl.sum()

    normalised_rel_dir_dist = np.divide(
        rel_dir_dist_div_ctrl,
        sum_rel_dir_dist_div_ctrl_ex,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where=sum_rel_dir_dist_div_ctrl_ex != 0
    )
    normalised_rel_dir_dist = normalised_rel_dir_dist * n_spikes_total

    return normalised_rel_dir_dist

# METHOD 1: FIND CONSINK
def find_consink(spike_train, reldir_occ_by_pos,  direction_bins,pos_data,
                 reldir_allframes, platforms_spk = None, verify_nans = True):
    """
    Find the consink position that maximises the mean resultant length of the normalised relative direction distribution.
    """
    if verify_nans:
        spike_train = np.array(spike_train)
        all_nans = verify_allnans(spike_train, pos_data)

        if all_nans:
            print("All nans. Returning nan")
            return np.nan, np.nan, np.nan

    if platforms_spk is None:
        # get head directions as np array
        platforms = pos_data['platform'].to_numpy()
        platforms_spk = platforms[spike_train]

        #mask = np.isnan(platforms_spk)
        #platforms_spk = platforms_spk[~mask]


    #  get control occupancy distribution
    rel_dir_ctrl_dist, n_spikes_total = rel_dir_ctrl_distribution_all_sinks(reldir_occ_by_pos,
                                                                            platforms_spk)

    # rel dir distribution for each possible consink position
    rel_dir_dist = rel_dir_distribution_all_sinks(spike_train, direction_bins, reldir_allframes)

    # normalise rel_dir_dist by rel_dir_ctrl_dist
    normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total)
    if np.isnan(normalised_rel_dir_dist).any():
        breakpoint()
    # calculate the mean resultant length of the normalised relative direction distribution
    mrl, mean_angles = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)
    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angles[max_mrl_indices[0][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle, mrl

############### METHOD 2 #################
# METHOD 2: SPIKE DISTRIBUTION

def rel_dir_distribution_m2_fast(spike_train, platforms_spk, reldir_bin_idx, n_bins):
    rel_dir_dist = np.zeros((num_candidate_sinks, 61, n_bins))
    bins = reldir_bin_idx[spike_train]  # (n_spikes, n_sinks)
    
    valid = ~np.isnan(platforms_spk)
    platforms_spk = platforms_spk[valid].astype(np.int64)
    bins = bins[valid]
    n_spikes_per_platform = np.bincount(platforms_spk - 1, minlength=61)
    for p in range(61):
        idx = np.where(platforms_spk == (p + 1))[0]
        if len(idx) == 0:
            continue
        
        # Relative bin for each spike to each sink for all spikes that occured on platform p
        bins_p = bins[idx]  # (n_spikes_p, n_sinks)

        for s in range(num_candidate_sinks):
            b = bins_p[:, s]
            b = b[b >= 0]
            rel_dir_dist[s, p] += np.bincount(b, minlength=n_bins)

    return rel_dir_dist, n_spikes_per_platform

def rel_dir_distribution_m2(spike_train, platforms_spk,  direction_bins, reldir_allframes):
    """
    METHOD 2 (normalize each platform)
    Create array to store the relative direction histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 3D array, with
    dimensions (n_sinks, n_platforms, n_direction_bins).

    """

    # create array to store relative direction histograms
    rel_dir_dist = np.zeros((num_candidate_sinks, 61, len(direction_bins) - 1))


    # n_spikes_per_platform
    n_spikes_per_platform = np.array([
        np.sum(platforms_spk == (p + 1)) for p in range(61)
    ])

    platform_masks = [(platforms_spk == (p + 1)) for p in range(61)]

    # loop through candidate consink positions
    for p in range(61):
        mask_p = platform_masks[p]
        
        for s in range(num_candidate_sinks):
            reldirections_sink = reldir_allframes[:, s]

            # get the relative direction
            relative_direction = reldirections_sink[spike_train]

            relative_direction_p = relative_direction[mask_p]
            relative_direction_p = relative_direction_p[~np.isnan(relative_direction_p)]

            if len(relative_direction_p) == 0:
                continue
            # bin the relative directions
            rel_dir_binned_counts, _ = bin_directions(relative_direction_p, direction_bins)
            rel_dir_dist[s, p, :] += rel_dir_binned_counts

    if np.all(rel_dir_dist == 0):
        print("rel dir dist all 0")

    return rel_dir_dist, n_spikes_per_platform

# METHOD 2: NORMALISATION
def normalize_rel_dir_dist_m2(rel_dir_dist, reldir_occ_by_pos,  n_spikes_per_platform):
    """
    Method 2 (normalize each platform)
    Normalise the relative direction distribution by the control distribution.
    """

    # first, divide rel_dir_dist by rel_dir_ctrl_dist
    rel_dir_dist_div_ctrl = np.divide(
        rel_dir_dist,
        reldir_occ_by_pos,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where= reldir_occ_by_pos != 0
    )

    # Step 2: normalize within each (sink, platform)
    # sum over direction bins
    sum_per_sink_platform = rel_dir_dist_div_ctrl.sum(axis=2)  # (n_sinks, n_platforms), gives the sum of the bins for each sink for each platform

    # expand dims for broadcasting
    sum_per_sink_platform_ex = sum_per_sink_platform[:, :, np.newaxis]

    rel_dir_dist_norm = np.divide(
        rel_dir_dist_div_ctrl,
        sum_per_sink_platform_ex,
        out=np.zeros_like(rel_dir_dist_div_ctrl),
        where=sum_per_sink_platform_ex != 0
    )

    # multiply by number of spikes on each platform
    rel_dir_dist_norm *= n_spikes_per_platform[np.newaxis, :, np.newaxis]

    # Step 3: sum across platforms
    rel_dir_dist_final = rel_dir_dist_norm.sum(axis=1)  # (n_sinks, n_dir_bins)

    return rel_dir_dist_final


# METHOD 2: FIND CONSINK
def find_consink_method2(spike_train, reldir_occ_by_pos,  direction_bins,pos_data,
                 reldir_allframes, reldir_bin_idx, platforms_spk = None, verify_nans = True):
    """ Here the firing for each platform is divided by the rel_dir_occ for that platform
    platfomrs_spk is not none for when we use this function to calculate the population sink"""
    spike_train = np.array(spike_train)

    if verify_nans:
        all_nans = verify_allnans(spike_train, pos_data)

        if all_nans:
            print("All nans. Returning nan")
            return np.nan, np.nan, np.nan

    if platforms_spk is None:
        platforms = pos_data['platform'].to_numpy()
        platforms_spk = platforms[spike_train]
        #mask = np.isnan(platforms_spk)
        #platforms_spk = platforms_spk[~mask]


    rel_dir_dist, n_spikes_per_platform = rel_dir_distribution_m2_fast(spike_train, platforms_spk, reldir_bin_idx, len(direction_bins) - 1)
    

    normalised_rel_dir_dist = normalize_rel_dir_dist_m2(rel_dir_dist, reldir_occ_by_pos,  n_spikes_per_platform)
    mrl, mean_angles = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)
    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angles[max_mrl_indices[0][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle, mrl

############### METHOD 3 #################
# METHOD 3: CONTROL DISTRIBUTION
def get_reldir_occ_wholemaze(reldir_allframes, direction_bins):
    """ From reldir_allframes, gets the directional occupancy for the whole maze (so not binned)"""
    
    reldir_occ_wholemaze = np.zeros((num_candidate_sinks, len(direction_bins) - 1))
    for s in range(num_candidate_sinks):
        relativedirections_sink = reldir_allframes[:, s] # getting all relative directions for s 
        relativedirections_sink = relativedirections_sink[~np.isnan(relativedirections_sink)]
        
        # bin the relative directions
        reldir_binned_counts, _ = bin_directions(relativedirections_sink, direction_bins)
        reldir_occ_wholemaze[s, :] = reldir_binned_counts
    return reldir_occ_wholemaze
        
         

# METHOD 3: NORMALISATION
def normalize_rel_dir_dist_m3(rel_dir_dist, reldir_occ_wholemaze):
    """
    Method 3 (normalize for the whole maze)
    Normalise the relative direction distribution by the control distribution.
    """

    # first, divide rel_dir_dist by rel_dir_ctrl_dist
    rel_dir_dist_final = np.divide(
        rel_dir_dist,
        reldir_occ_wholemaze,
        out=np.zeros_like(rel_dir_dist, dtype=float),
        where= reldir_occ_wholemaze != 0
    )
    return rel_dir_dist_final

# METHOD 3: FIND CONSINK
def find_consink_method3(spike_train, reldir_occ_wholemaze,  direction_bins,pos_data,
                 reldir_allframes, platforms_spk = None, verify_nans = True):
    """ Here the rel_dir_dist is divided by the occupancy for the hwole maze"""
    spike_train = np.array(spike_train)

    if verify_nans:
        all_nans = verify_allnans(spike_train, pos_data)

        if all_nans:
            print("All nans. Returning nan")
            return np.nan, np.nan, np.nan

    if platforms_spk is None:
        platforms = pos_data['platform'].to_numpy()
        platforms_spk = platforms[spike_train]
        #mask = np.isnan(platforms_spk)
        #platforms_spk = platforms_spk[~mask]

    rel_dir_dist= rel_dir_distribution_all_sinks(spike_train, direction_bins, reldir_allframes)
    normalised_rel_dir_dist = normalize_rel_dir_dist_m3(rel_dir_dist, reldir_occ_wholemaze)
    mrl, mean_angles = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)
    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angles[max_mrl_indices[0][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle, mrl


############# SIGNIFICANCE TESTING ###############

# Shifting spike train
def shift_spiketrain_pergoal(spike_train, goal, intervals_frames, n_frames: int, frame_rate=25):
    """Shift the spike train by a random amount. Restrict it to goal intervals

    Args:
        spike_train (array): firing times of unit (in frames)
        goal(int): goal number
        n_frames (int): length of the recording (in frames)
    """
    if goal < 3:
        start_col = goal * 2
        end_col = goal*2 + 1
    else:
        start_col = 0
        end_col = intervals_frames.shape[1] - 1
    spike_train = np.array(spike_train)
    min_shift = 2 * frame_rate
    lengths = [intervals_frames.iloc[tr, end_col] - intervals_frames.iloc[tr, start_col] for tr in
               range(len(intervals_frames))]
    max_shift = np.max(lengths) - min_shift + 1


    # pick a shift randomly between those two numbers
    shift = np.random.randint(min_shift, max_shift)

    shifted_data = np.empty(0, dtype=int)

    for tr in range(len(intervals_frames)):
        start_frame = intervals_frames.iloc[tr, start_col]
        end_frame = intervals_frames.iloc[tr, end_col]
        spike_train_tr = spike_train[(spike_train >= start_frame) & (spike_train <= end_frame)]

        if len(spike_train_tr) == 0:
            continue
        shifted_data_tr = spike_train_tr + shift

        range_min = start_frame
        range_max = end_frame
        range_size = range_max - range_min + 1

        shifted_data_tr = np.mod(shifted_data_tr - range_min, range_size) + range_min

        if np.min(shifted_data_tr) < range_min or np.max(shifted_data_tr) > range_max:
            breakpoint()

        shifted_data = np.append(shifted_data, shifted_data_tr)

    shifted_data = np.array(shifted_data)

    return shifted_data


# Calculate MRL for shifted spike train
def calculate_translated_mrl(spiketrain, dlc_data, reldir_occ_by_pos,  direction_bins,
                             reldir_allframes, reldir_occ_wholemaze, intervals_frames, reldir_bin_idx, method, goal):
    n_frames = len(dlc_data)
    translated_spiketrain = shift_spiketrain_pergoal(spiketrain, goal, intervals_frames, n_frames)

    if method == 1:
        mrl, *_ = find_consink(translated_spiketrain, reldir_occ_by_pos,  direction_bins,
                                 dlc_data, reldir_allframes)
    elif method ==2:
        mrl, *_ = find_consink_method2(translated_spiketrain, reldir_occ_by_pos, direction_bins,
                                 dlc_data, reldir_allframes, reldir_bin_idx)
    elif method == 3:
        mrl, *_ = find_consink_method3(translated_spiketrain, reldir_occ_wholemaze, direction_bins,
                                 dlc_data, reldir_allframes)
    else:
        raise ValueError("Method must be 1, 2 or 3")
    return mrl

# Do shuffling
def recalculate_consink_to_all_candidates_from_translation(spiketrain, dlc_data, reldir_occ_by_pos,
                                                           direction_bins,reldir_allframes,
                                                           reldir_occ_wholemaze, intervals_frames, reldir_bin_idx, method, goal):
    n_shuffles = 1000
    mrl = np.zeros(n_shuffles)

    # mrl = Parallel(n_jobs=-1, verbose=0)(delayed(calculate_translated_mrl)(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles))
    mrl = Parallel(n_jobs=-1, verbose=0)(
        delayed(calculate_translated_mrl)(spiketrain, dlc_data, reldir_occ_by_pos,  direction_bins,
                                           reldir_allframes, reldir_occ_wholemaze, intervals_frames,reldir_bin_idx,  method, goal)
        for s in range(n_shuffles))



    # mrl = [calculate_translated_mrl(spiketrain, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles)]
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
    ci = (mrl_95, mrl_999, n_shuffles - len(mrl))  # last one is the length after nans are removed

    return ci

############ AVERAGE SINK ############
def calculate_averagesink(consinks_df, hcoord, vcoord, goals_to_include):
    average_positions = {}
    for g in goals_to_include:

        position = []
        for cluster in consinks_df.index:
            sig = consinks_df.loc[cluster, 'sig_g' + str(g)]
            if sig == "sig":

                consink_plat = consinks_df.loc[cluster, 'platform_g' + str(g)]
                position.append([hcoord[np.int32(consink_plat)- 1], vcoord[np.int32(consink_plat) - 1]])
        # make the axes equal
        avg_pos = np.mean(position, axis = 0)
        average_positions[g] = avg_pos
    return average_positions