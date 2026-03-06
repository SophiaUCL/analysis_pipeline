import matplotlib
import numpy as np
import pandas as pd
import os
matplotlib.use("TkAgg")
import pickle
import datetime
import spikeinterface.extractors as se

from astropy.stats import circmean
from HCT_analysis.consinks.RelDirOcc_functions import get_directions_to_position, get_relative_directions_to_position, get_relative_direction_occupancy_by_position_platformbins
from HCT_analysis.consinks.find_consinks_main_functions import mean_resultant_length_nrdd
from HCT_analysis.utilities.platforms_utils import get_platform_center, calculate_occupancy_plats, get_hd_distr_allplats, \
from HCT_analysis.utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from HCT_analysis.utilities.trials_utils import get_direction_bins, bin_directions, get_unit_ids




def load_directories(derivatives_base, dirname = 'popsinks'):
    """ Loads directories that we need and pos data"""
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)

    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting", "sorter_output")
    sorting = se.read_kilosort(
        folder_path=kilosort_output_path
    )

    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                 'XY_HD_w_platforms.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])

    pos_data_path_intervals = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                           'XY_HD_allintervals.csv')
    pos_data_iv = pd.read_csv(pos_data_path_intervals)

    if np.nanmax(pos_data_iv['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
        pos_data_iv['hd'] = np.deg2rad(pos_data_iv['hd'])

    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features',
                                 dirname)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return rawsession_folder, sorting, pos_data, pos_data_iv, output_folder


def restrict_pos_data(derivatives_base, g, pos_data_iv):
    """ Returns positional data used for the analaysis for goal g
    if g < 3 (meaning g0, 1 or 2), we only load positional data for that goal
    if g == 3, we use the data for the full interval"""
    if g < 3:
        pos_data_path_g = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                       f'XY_HD_goal{g}_trials.csv')
        pos_data_g = pd.read_csv(pos_data_path_g)
        platform_occupancy_g = calculate_occupancy_plats(pos_data_g)  # platform occupancy in frames
        hd_distr_g, bin_centers = get_hd_distr_allplats(pos_data_g)
    else:
        pos_data_g = pos_data_iv
        platform_occupancy_g = calculate_occupancy_plats(pos_data_iv)
        hd_distr_g, bin_centers = get_hd_distr_allplats(pos_data_iv)
    return pos_data_g, platform_occupancy_g, hd_distr_g, bin_centers


def get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, pos_data):
    """ Gets the spiketrain for unit unit_id for goal g. Spiketrain returned is in frames
    If g < 3 (so g0, 1 or 2), we restrict the spiketrain to only that goal"""

    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_secs = np.round(
        spike_train_unscaled / sample_rate)  # trial data is now in frames in order to match it with xy data

    # If we're only looking at one goal, restrict the spiketrain to match only period of goal 1 or 2
    if g < 3:
        spike_train_restricted = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder=rawsession_folder,
                                                                  goal=g)
    else:
        spike_train_restricted = spike_train_secs
    spike_train = spike_train_restricted * frame_rate
    spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]
    return spike_train


def get_norm_mean_angle(norm_hd_distr, bin_centers):
    """ Returns mean angle for each platform. If all values are nan or 0, returns nan"""
    norm_mean_angle = []
    for i in range(61):
        if np.all(np.isnan(norm_hd_distr[i])) or np.all(norm_hd_distr[i] == 0):
            norm_mean_angle.append(np.nan)
        else:
            norm_mean_angle.append(circmean(bin_centers, weights=norm_hd_distr[i]))
    return norm_mean_angle


def get_data_popsink_calc(hcoord, vcoord, mean_norm_vecs_angle, mean_norm_vecs_length):
    """ This data will be used for consink calculation"""
    pos_org = np.array(
        [get_platform_center(hcoord, vcoord, p + 1) for p in range(61)])  # positions are the center for each platform
    plats_org = np.arange(1, 62)
    hd_org = mean_norm_vecs_angle  # hd is the mean angle for each platform
    nspikes_org = np.round(np.array(mean_norm_vecs_length) * 1000).astype(int)
    return pos_org, plats_org, hd_org, nspikes_org


def remove_nan(pos_org, plats_org, hd_org, nspikes_org, indices_to_del):
    """ Removes nans from the data used for consink calculation"""
    plats = np.delete(plats_org, indices_to_del)
    pos = np.delete(pos_org, indices_to_del, axis=0)
    hd = np.delete(hd_org, indices_to_del)
    nspikes = np.delete(nspikes_org, indices_to_del)
    if len(plats) != len(hd) or len(plats) != len(pos) or len(hd) != len(pos) or len(nspikes) != len(pos):
        raise ValueError("Lengths of plats, hd, and pos do not match after removing NaNs.")
    return pos, plats, hd, nspikes


def create_spikedata(pos, hd, plats, nspikes):
    """ Repeats each value by nspikes"""
    spikePos = np.repeat(pos, nspikes, axis=0)
    spikeHD = np.repeat(hd, nspikes)
    spikePlats = np.repeat(plats, nspikes)
    return spikePos, spikeHD, spikePlats


def get_dir_allframes(spikePos, spikeHD, sink_positions, num_candidate_sinks = 127):
    """ Gets directions from each frame to each sink"""

    sinkdir_allframes = np.zeros(
        (len(spikePos), num_candidate_sinks)
    )

    reldir_allframes = np.zeros(
        (len(spikePos), num_candidate_sinks)
    )


    x_org, y_org = spikePos.T

    positions = {'x': x_org, 'y': y_org}
    for s in range(num_candidate_sinks):
        platform_loc = sink_positions[s]
        directions = get_directions_to_position([platform_loc[0], platform_loc[1]], positions)
        sinkdir_allframes[:, s] = directions
        relative_direction = get_relative_directions_to_position(directions, spikeHD)
        reldir_allframes[:, s] = relative_direction
    return sinkdir_allframes, reldir_allframes



def save_session_data(data_folder, method, pos, hd, nspikes, pos_org, platform_occupancy_g, scaled_vecs_allplats, allscales,
                      allfiring_rates, allnorm_MRL, hcoord, vcoord, unit_type, hd_org, nspikes_org, g, relDirDist):
    """ Saves session data"""
    session_data = {
        'pos': pos,
        'hd': hd,
        'nspikes': nspikes,
        'pos_org': pos_org,
        'hd_org': hd_org,
        'nspikes_org': nspikes_org,
        'occupancy': platform_occupancy_g,
        'scaled_vecs_allplats': scaled_vecs_allplats,
        'allscales': allscales,
        'allfiring_rates': allfiring_rates,
        'allnorm_MRL': allnorm_MRL,
        'hcoord': hcoord,
        'vcoord': vcoord,
        'timestamp': datetime.datetime.now().isoformat(),
        'goal': g,
        'unit_type': unit_type,
        'relDirDist': relDirDist,
        'comment': 'Intermediate population sink data for verification and reuse.'
    }

    # Save the dictionary as a pickle
    session_filename = os.path.join(data_folder, f'session_vars_goal{g}_{unit_type}_{method}.pkl')
    with open(session_filename, 'wb') as f:
        pickle.dump(session_data, f)

    print(f"✅ Saved session variables to {session_filename}")


def get_dirdist_from_reldir_allframes(reldir_allframes, direction_bins, num_candidate_sinks = 127):

    dir_dist = np.zeros((num_candidate_sinks,len(direction_bins) - 1))

    for s in range(num_candidate_sinks):
        reldirections_sink = reldir_allframes[:, s]
        rel_dir_binned_counts, _ = bin_directions(reldirections_sink, direction_bins)
        dir_dist[s, :] += rel_dir_binned_counts

    return dir_dist

def find_popsink_m3(reldir_allframes, direction_bins):
    print("Entered m3")
    dir_dist = get_dirdist_from_reldir_allframes(reldir_allframes, direction_bins)

    mrl, mean_angles = mean_resultant_length_nrdd(dir_dist, direction_bins)
    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0
    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angles[max_mrl_indices[0][0]],3)
    return np.round(max_mrl, 3), max_mrl_indices, mean_angle, dir_dist
