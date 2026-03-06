from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import numpy as np
import os
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from utilities.spatial_functions import get_ratemaps, get_ratemaps_restrictedx
import json

from consinks.plot_sinks import plot_consinks_singlesubplot
from utilities.trials_utils import get_goal_numbers, get_coords_127sinks, get_pos_data,get_sink_positions_platforms, translate_positions
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from consinks.RelDirOcc_functions import get_relative_direction_occupancy_by_position_platformbins
from tqdm import tqdm
from consinks.find_consinks_main_functions import get_reldir_bin_idx, find_consink, get_reldir_occ_wholemaze, recalculate_consink_to_all_candidates_from_translation, find_consink_method2, find_consink_method3, get_dir_allframes

""" Allows you to interactively select an area on a rmap and then plot the sinks for that area"""

def mask_posdata(pos_data, mask):
    """
    Masks positional data

    Args:
        pos_data: array with x, y, hd, x_bin and y_bin
        mask: 2D boolean
    returns hd_masked
    """
    xsize, ysize = mask.shape

    x_bin = pos_data["x_bin"].to_numpy()
    y_bin = pos_data["y_bin"].to_numpy()
    hd = pos_data["hd"].to_numpy()

    valid = (
            (x_bin >= 0) & (x_bin < xsize) &
            (y_bin >= 0) & (y_bin < ysize)
    )

    valid_mask = np.zeros_like(valid, dtype=bool)
    valid_indices = np.where(valid)
    valid_mask[valid_indices] = mask[x_bin[valid], y_bin[valid]]

    # Return masked hd values (ignore NaNs)
    return hd[valid_mask & ~np.isnan(hd)]

def load_data(derivatives_base, include_g0):
    """ Loads all the data"""
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, 'ephys', "concat_run", "sorting", "sorter_output")
    sorting = se.read_kilosort(
        folder_path=kilosort_output_path
    )

    # Limits
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()

    xmin = limits['x_min']
    xmax = limits['x_max']
    ymin = limits['y_min']
    ymax = limits['y_max']

    # ---- Load maze outline coordinates ----
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    if os.path.exists(outline_path):
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        print("⚠️ Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None



    input = 'c'
    mask = None
    goals = [0,1,2,3] if include_g0 else [1,2, 3] # goals == 3: full trials (not restricted to any of the goals)
    return xmin, xmax, ymin, ymax, outline_x, outline_y, input, mask, sorting, goals

def get_spiketrain_allgoals(rawsession_folder, unit_id, sorting, goals, pos_data, frame_rate = 25, sample_rate = 30000):
    """ Gets spiketrain for all the goals"""
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_secs = spike_train_unscaled / sample_rate  # This is in seconds now

    spiketrains = {}

    for g in goals:
        if g != 3: # g == 3, full trial
            spike_train_secs_g = restrict_spiketrain_specialbehav(
                spike_train_secs, rawsession_folder, goal=g
            )
        else:
            spike_train_secs_g = spike_train_secs
        spike_train = np.round(spike_train_secs_g * frame_rate)
        spike_train = np.array(
            [el for el in spike_train if el < len(pos_data)],
            dtype=np.int32
        )

        spiketrains[g] = spike_train

    return spiketrains

def get_ratemaps_allgoals(spiketrains, goals, pos_datasets):
    """ Gets ratemaps for all goals"""
    rmaps = {}
    pos_data = pos_datasets[3]
    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()

    for g in goals:
        spike_train = spiketrains[g]
        pos_data_g = pos_datasets[g]

        if g ==  3:
            rmap, x_edges, y_edges = get_ratemaps(spike_train, x, y, 3, binsize=36, stddev=25)
        else:
            x_restr = pos_data.iloc[:, 0].to_numpy()
            y_restr = pos_data.iloc[:, 1].to_numpy()
            rmap, x_edges, y_edges = get_ratemaps_restrictedx(spike_train, x, y, x_restr, y_restr,3, binsize=36, stddev=25)
        rmaps[g] = rmap
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1

    mask_x = np.isnan(x)
    mask_y = np.isnan(y)

    y_bin[mask_y] = -1
    x_bin[mask_x] = -1

    pos_data['x_bin'] = x_bin
    pos_data['y_bin'] = y_bin

    return rmaps, pos_data, x_bin, y_bin, x_edges, y_edges

def get_filtered_spiketrain(spike_train, mask, x_bin, y_bin):
    """ Returns values in spike_train that are within the masked region"""
    if mask is not None:
        xsize, ysize = mask.shape
        spike_train_filt = []
        spike_train_outside = []
        for s in spike_train:
            indx = x_bin[s]
            indy = y_bin[s]
            if indx < xsize and indy < ysize and mask[indx, indy]:
                spike_train_filt.append(s)
            else:
                spike_train_outside.append(s)

    else:
        spike_train_filt = spike_train
        spike_train_outside = []

    is_filt = np.isin(spike_train, spike_train_filt)
    return spike_train_filt, spike_train_outside, is_filt

def plot_rmap(rmap, i, x_edges, y_edges, xmin, xmax, ymin, ymax, outline_x,outline_y, fig, axs, title):
    """ PLots ratemap"""
    im = axs[i, 0].imshow(rmap.T,
                       cmap='viridis',
                       interpolation=None,
                       origin='lower',
                       aspect='auto',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])

    axs[i, 0].set_title(title)
    axs[i, 0].set_xlim(xmin, xmax)
    axs[i, 0].set_ylim(ymax, ymin)
    axs[i, 0].set_aspect('equal')
    if outline_x is not None and outline_y is not None:
        axs[i, 0].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    fig.colorbar(im, ax=axs[i, 0], label='Firing rate')

def get_data_for_consinks(derivatives_base, pos_data, pos_data_reldir, pos_data_g0, pos_data_g1, pos_data_g2, include_g0):
    """ Getting all the data we need for the consink calculation"""
    platforms_trans = translate_positions()
    direction_bins = get_direction_bins(n_bins=12)
    sink_positions = get_sink_positions_platforms(derivatives_base)
    goal_numbers = get_goal_numbers(derivatives_base)
    sinkdir_allframes, reldir_allframes = get_dir_allframes(pos_data, sink_positions)

    reldir_occ_by_pos = get_relative_direction_occupancy_by_position_platformbins(pos_data_reldir, sink_positions,
                                                                                  num_candidate_sinks=127,
                                                                                  n_dir_bins=12, frame_rate=25)
    if include_g0:
        reldir_occ_by_pos_g0 = get_relative_direction_occupancy_by_position_platformbins(pos_data_g0, sink_positions,
                                                                                     num_candidate_sinks=127,
                                                                                     n_dir_bins=12, frame_rate=25)
    else:
        reldir_occ_by_pos_g0 = None
    reldir_occ_by_pos_g1 = get_relative_direction_occupancy_by_position_platformbins(pos_data_g1, sink_positions,
                                                                                     num_candidate_sinks=127,
                                                                                     n_dir_bins=12, frame_rate=25)
    reldir_occ_by_pos_g2 = get_relative_direction_occupancy_by_position_platformbins(pos_data_g2, sink_positions,
                                                                                     num_candidate_sinks=127,
                                                                                     n_dir_bins=12, frame_rate=25)
    return platforms_trans, direction_bins, goal_numbers, reldir_allframes, reldir_occ_by_pos, reldir_occ_by_pos_g0, reldir_occ_by_pos_g1, reldir_occ_by_pos_g2

def plot_spikemap(i, spike_train, x, y, hd, xmin, xmax, ymin, ymax, outline_x, outline_y, is_filt, axs,  title):
    """ Makes spikemap plot"""
    x_spikes = x[spike_train]
    y_spikes = y[spike_train]
    hd_spikes = hd[spike_train]

    # Only valid values
    valid = ~np.isnan(x_spikes) & ~np.isnan(y_spikes) & ~np.isnan(hd_spikes)
    x_spikes = x_spikes[valid]
    y_spikes = y_spikes[valid]
    hd_spikes = hd_spikes[valid]
    is_filt = is_filt[valid]

    u = np.cos(hd_spikes)
    v = np.sin(hd_spikes)

    # Assign colors efficiently
    colors = np.where(is_filt, 'blue', 'red')

    # Plot
    axs[i, 1].quiver(x_spikes, y_spikes, u, v, color=colors, scale=30)
    axs[i, 1].set_xlim(xmin, xmax)
    axs[i, 1].set_ylim(ymax, ymin)
    axs[i, 1].set_aspect('equal')
    if outline_x is not None and outline_y is not None:
        axs[i, 1].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    axs[i, 1].set_title(title)


def plot_consinks_interactive(derivatives_base, unit_id, methods = [1,2,3], rel_dir_occ = "intervals",  goals: list = [0,1,2],  frame_rate=25, sample_rate=30000):
    """
    Makes a plot for each unit with its ratemap (left), occupancy (middle) and directional firing rate (right).
    User adjustable

    Inputs: derivatives base

    """
    # Path to rawsession folder
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)

    # restricted df frames
    path = os.path.join(rawsession_folder, 'task_metadata', 'restricted_df_frames.csv')
    intervals_frames = pd.read_csv(path)
    
    # Loading data
    xmin, xmax, ymin, ymax, outline_x, outline_y,input, mask, sorting, goals = load_data(derivatives_base, include_g0)
    goals = [1,2]
    pos_data, pos_data_g0, pos_data_g1, pos_data_g2, pos_data_reldir = get_pos_data(derivatives_base, rel_dir_occ)

    pos_datasets = {
        0: pos_data_g0,
        1: pos_data_g1,
        2: pos_data_g2,
        3: pos_data
    }

    x = pos_data.iloc[:, 0].to_numpy()
    y = pos_data.iloc[:, 1].to_numpy()
    hd = pos_data.iloc[:, 2].to_numpy()

    hcoord, vcoord = get_coords_127sinks(derivatives_base)
    x_diff = np.mean(np.diff(hcoord))

    y_diff = np.mean(np.diff(vcoord))
    jitter = (2*x_diff, 2*y_diff)


    spiketrains = get_spiketrain_allgoals(rawsession_folder, unit_id, sorting, goals, pos_data)

    rmaps, pos_data, x_bin, y_bin, x_edges, y_edges = get_ratemaps_allgoals(spiketrains, goals,  pos_datasets)

    platform_trans, direction_bins, goal_numbers, reldir_allframes, reldir_occ_by_pos, reldir_occ_by_pos_g0, reldir_occ_by_pos_g1, reldir_occ_by_pos_g2 = get_data_for_consinks(derivatives_base, pos_data, pos_data_reldir, pos_data_g0, pos_data_g1, pos_data_g2, include_g0)
    reldir_occ_wholemaze = get_reldir_occ_wholemaze(reldir_allframes, direction_bins)
    reldir_bin_idx = get_reldir_bin_idx(reldir_allframes, direction_bins)
    rmap_axes = {}
    # Rel dir occ
    whole_field_firing = {
        1: {},
        2: {},
        3: {}
    }
    while input != 'q':
        # Make plot
        nrows = len(goals)
        ncols = 2 + len(methods)  # Ratemap, spikemap, and one for each method
        fig, axs = plt.subplots(len(goals), 2 + len(methods), figsize=[nrows*5, ncols*5], constrained_layout=True)
        fig.suptitle(f"Unit {unit_id}", fontsize=18)

        for i, g in tqdm(enumerate(goals)): # We make separate plot for each of them
            spike_train = spiketrains[g]
            rmap = rmaps[g]
            rmap_axes[g] = axs[i, 0]
            spike_train_filt,spike_train_outside,  is_filt = get_filtered_spiketrain(spike_train, mask, x_bin, y_bin)


            if g == 0:
                title = "Going to g2 during g1"
                reldir_occ_by_pos_cur = reldir_occ_by_pos_g0
            elif g < 3:
                title = f"Goal {g}"
                if g == 1:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                else:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
            else:
                reldir_occ_by_pos_cur = reldir_occ_by_pos
                title = "Full trial"
            # First plot: ratemap
            plot_rmap(rmap, i, x_edges, y_edges, xmin, xmax, ymin, ymax, outline_x,outline_y, fig, axs, title = title)

            # Second plot: spikemap
            plot_spikemap(i, spike_train, x, y, hd, xmin, xmax, ymin, ymax, outline_x, outline_y, is_filt, axs, title =f"{len(spike_train_filt)}/{len(spike_train)}" )

            # Third, consink plots
            # For within
            for m in methods: # separating for method 1 and method 2 and 3
                if g not in whole_field_firing[m]:
                    whole_field_firing[m][g] = {}
                consink_plat = []
                consink_sig = []
                max_mrls = []
                mean_angles = []
                percentile_95th = []

                if m == 1:
                        max_mrl, max_mrl_indices, mean_angle = find_consink(
                            spike_train_filt, reldir_occ_by_pos_cur, direction_bins, pos_data,
                            reldir_allframes
                        )
                elif m == 2:
                    max_mrl, max_mrl_indices, mean_angle = find_consink_method2(
                        spike_train_filt, reldir_occ_by_pos_cur, direction_bins, pos_data,
                        reldir_allframes, reldir_bin_idx
                    )
                elif m == 3:
                    max_mrl, max_mrl_indices, mean_angle = find_consink_method3(
                        spike_train_filt, reldir_occ_wholemaze, direction_bins, pos_data,
                        reldir_allframes
                    )
                consink_plat.append(max_mrl_indices[0][0] + 1)
                max_mrls.append(max_mrl)
                mean_angles.append(mean_angle)

                ci = recalculate_consink_to_all_candidates_from_translation(spike_train_filt, pos_data,
                                                                                reldir_occ_by_pos_cur,
                                                                                direction_bins,
                                                                                reldir_allframes, reldir_occ_wholemaze,
                                                                                intervals_frames, reldir_bin_idx, method =m, goal=g)

                if np.isfinite(ci[0]) and np.isfinite(max_mrl) and max_mrl > ci[0]:
                    consink_sig.append("sig")
                else:
                    consink_sig.append("ns")
                percentile_95th.append(ci[0])
                # For the outside
                if mask is not None:
                    if m == 1:
                        max_mrl, max_mrl_indices, mean_angle = find_consink(
                            spike_train_outside, reldir_occ_by_pos_cur, direction_bins, pos_data,
                            reldir_allframes
                        )
                    elif m == 2:
                        max_mrl, max_mrl_indices, mean_angle = find_consink_method2(
                        spike_train_outside, reldir_occ_by_pos_cur, direction_bins, pos_data,
                        reldir_allframes, reldir_bin_idx
                    )
                    elif m == 3:
                        max_mrl, max_mrl_indices, mean_angle = find_consink_method3(
                        spike_train_outside, reldir_occ_wholemaze, direction_bins, pos_data,
                        reldir_allframes
                     )
                    consink_plat.append(max_mrl_indices[0][0] + 1)

                    ci = recalculate_consink_to_all_candidates_from_translation(spike_train_outside, pos_data,
                                                                                reldir_occ_by_pos_cur,
                                                                                direction_bins,
                                                                                reldir_allframes, reldir_occ_wholemaze,
                                                                                intervals_frames, reldir_bin_idx, method =m, goal=g)

                    if np.isfinite(ci[0]) and np.isfinite(max_mrl) and max_mrl > ci[0]:
                        consink_sig.append("sig")
                    else:
                        consink_sig.append("ns")
                    max_mrls.append(max_mrl)
                    mean_angles.append(mean_angle)
                    percentile_95th.append(ci[0])
                else: # if mask is None, so appended in the first round only
                    whole_field_firing[m][g]['max_mrl'] = max_mrl
                    whole_field_firing[m][g]['mean_angle'] = mean_angle
                    whole_field_firing[m][g]['percentile_95th'] = ci[0]
                    whole_field_firing[m][g]['consink_plat'] = max_mrl_indices[0][0] + 1

                title = 'Normalised by trial behaviour' if m == 1 else 'Normalised by platform behaviour'
                plot_consinks_singlesubplot(consink_plat,consink_sig, max_mrls, mean_angles, percentile_95th, i,  g, goal_numbers, hcoord, vcoord, platform_trans, whole_field_firing[m][g], jitter, title = title, ax = axs[i, 1 + m])





        # Now let the user draw on the ratemap (axs[0])
        #mask = select_region(rmap, axs[0, 0], x_edges, y_edges)
        selected_goal, mask = enable_overlay_selection(
            rmaps, rmap_axes, x_edges, y_edges
        )
        inside_coords = np.argwhere(mask)
        print("Number of points inside polygon:", inside_coords.shape[0])


def enable_overlay_selection(rmaps, rmap_axes, x_edges, y_edges):
    """
    Enables polygon selection on any ratemap.
    Returns (selected_goal, mask)
    """

    result = {"goal": None, "mask": None}

    def make_onselect(goal, rmap, ax):
        def onselect(verts):
            ny, nx = rmap.shape
            x_lin = np.linspace(x_edges[0], x_edges[-1], ny)
            y_lin = np.linspace(y_edges[0], y_edges[-1], nx)
            X, Y = np.meshgrid(x_lin, y_lin)

            points = np.vstack((X.ravel(), Y.ravel())).T
            path = Path(verts)
            mask = path.contains_points(points).reshape((nx, ny)).T

            ax.contour(mask, colors='r', linewidths=0.8)
            plt.draw()

            result["goal"] = goal
            result["mask"] = mask

            # Disable all selectors once one is used
            for sel in selectors.values():
                sel.disconnect_events()
            plt.close()

        return onselect

    selectors = {}

    for g, ax in rmap_axes.items():
        selectors[g] = PolygonSelector(
            ax,
            make_onselect(g, rmaps[g], ax),
            props=dict(color='r', linewidth=2, alpha=0.6)
        )

    print("Draw a polygon on ANY ratemap (double-click to finish)...")
    plt.show(block=True)

    return result["goal"], result["mask"]

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    unit_id = 5
    plot_consinks_interactive(derivatives_base, unit_id, include_g0 = False)


