from pathlib import Path
import sys
import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from HCT_analysis.calculate_occupancy import get_direction_bins, \
     get_relative_direction_occupancy_by_position_platformbins
from HCT_analysis.utilities.load_and_save_data import load_pickle, save_pickle
from HCT_analysis.utilities.trials_utils import ensure_sig_columns, get_goal_numbers, get_coords_127sinks, get_unit_ids, get_pos_data, get_spike_train, get_sink_positions_platforms, translate_positions
import matplotlib
from HCT_analysis.turn_restricteddf_frames import turn_restricteddf_frames
from HCT_analysis.plotting.plot_sinks import  plot_all_consinks_127sinks
from HCT_analysis.find_consinks_main_functions import get_reldir_bin_idx, calculate_averagesink, find_consink, get_reldir_occ_wholemaze, recalculate_consink_to_all_candidates_from_translation, find_consink_method2, find_consink_method3, get_dir_allframes
from maze_and_platforms.overlay_maze_image_consinks import overlay_maze_image_consinks
num_candidate_sinks = 127
from astropy.stats import circmean
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize
import matplotlib.cm as cm
matplotlib.use("QtAgg")
from tqdm import tqdm
from typing import Literal


""" In this code, the bins are the platforms (127 in total)
We can use three methods for the consink calculation:
Method 1: normalise the relative direction distribution by the control distribution based on the number of spikes fired at each platform (original method)
Method 2: normalise the relative direction distribution for each platform separately, then sum across platforms
Method 3: normalise the relative direction distribution by the control distribution based on the occupancy for the whole maze (not binned by platform)

NOTE: Potential issues to still look at: if ctrl distribution has low values in certain bin, it will inflate the relative direction occupancy for that bin. 
This can be fixed by setting a threshold for min occupancy in ctrl distribution
Threshold hasn't been decided yet
"""
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

def main(derivatives_base, rel_dir_occ: Literal['all trials', 'intervals'],
         unit_type: Literal['pyramidal', 'good', 'all', 'test'], methods = [1,2,3],  code_to_run=[-1, 0, 1,2,3,4], goals_to_include = [0,1,2], show_plots = True, frame_rate=25, sample_rate=30000):
    """
    Code to find consinks, based on Jake's code


    """
    print(f"Calculating consinks using methods {methods}")
    # Path to rawsession folder
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)

   
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting", "sorter_output")
    sorting = se.read_kilosort(
        folder_path=kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    unit_ids = get_unit_ids(derivatives_base, unit_ids, unit_type)

    # Loading xy data
    pos_data, pos_data_g0, pos_data_g1, pos_data_g2, pos_data_reldir = get_pos_data(derivatives_base, rel_dir_occ)

    # restricted df frames
    path = os.path.join(rawsession_folder, 'task_metadata', 'restricted_df_frames.csv')
    if not os.path.exists(path):
        turn_restricteddf_frames(derivatives_base, frame_rate=frame_rate)
    intervals_frames = pd.read_csv(path)

    if not os.path.exists(os.path.join(derivatives_base, 'analysis', 'maze_overlay', 'maze_overlay_params_consinks.json')):
        overlay_maze_image_consinks(derivatives_base, method="video")
        
    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features',
                                 'consinks')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Direction bins (from -pi to pi)
    direction_bins = get_direction_bins(n_bins=12)

    # gets translated positions
    platforms_trans = translate_positions()
    # Loading or creating data
    sink_positions = get_sink_positions_platforms(derivatives_base)
    goal_numbers= get_goal_numbers(derivatives_base)
    _, reldir_allframes = get_dir_allframes(pos_data, sink_positions)
    reldir_occ_wholemaze = get_reldir_occ_wholemaze(reldir_allframes, direction_bins)
    reldir_bin_idx = get_reldir_bin_idx(reldir_allframes, direction_bins)
    file_name = 'reldir_occ_by_pos.npy'
    min_num_spikes = 30
    if -1 in code_to_run:
        print("Calculating relative direction occupancy by position")
        reldir_occ_by_pos= get_relative_direction_occupancy_by_position_platformbins(pos_data_reldir, sink_positions,num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)
        np.save(os.path.join(output_folder, file_name), reldir_occ_by_pos)
        if 0 in goals_to_include:
            reldir_occ_by_pos_g0= get_relative_direction_occupancy_by_position_platformbins(pos_data_g0, sink_positions,num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)
            np.save(os.path.join(output_folder, 'reldir_occ_by_pos_g0.npy'), reldir_occ_by_pos_g0)
        if 1 in goals_to_include:
            reldir_occ_by_pos_g1 = get_relative_direction_occupancy_by_position_platformbins(pos_data_g1, sink_positions,num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)
            np.save(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy'), reldir_occ_by_pos_g1)
        if 2 in goals_to_include:
            reldir_occ_by_pos_g2 = get_relative_direction_occupancy_by_position_platformbins(pos_data_g2, sink_positions,num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)
            np.save(os.path.join(output_folder, 'reldir_occ_by_pos_g2.npy'), reldir_occ_by_pos_g2)

    else:
        print("Loading reldir occ, not calculating")
        if 0 in goals_to_include:
            reldir_occ_by_pos_g0= np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g0.npy'))
        reldir_occ_by_pos = np.load(os.path.join(output_folder, file_name))
        if 1 in goals_to_include:
            reldir_occ_by_pos_g1 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy'))
        if 2 in goals_to_include:
            reldir_occ_by_pos_g2 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g2.npy'))

    ################# CALCULATE CONSINKS ###########################################
    
    for method in methods:
        # Constructing title

        subject_id = derivatives_base.split(os.sep)[3]
        session_id = derivatives_base.split(os.sep)[4]
        title = f"Consinks {subject_id}_{session_id}, {unit_type} units, method {method}"
        consinks = {}

        if 0 in code_to_run:
            print("Calculating consinks")
            for unit_id in tqdm(unit_ids):
                consinks[unit_id] = {'unit_id': unit_id}

                for g in goals_to_include:
                    if g == 0:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos_g0
                    if g == 1:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                    elif g == 2:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
                    else:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos


                    spike_train = get_spike_train(sorting, unit_id, pos_data, rawsession_folder, g=g, frame_rate=frame_rate,
                                                sample_rate=sample_rate)
                    # Skip empty spikes
                    if len(spike_train) < min_num_spikes:
                        consinks[unit_id][f'numspikes_g{g}'] = len(spike_train)
                        continue

                    # get consink
                    if method == 1:
                        max_mrl, max_mrl_indices, mean_angle, mrl_values = find_consink(
                            spike_train, reldir_occ_by_pos_cur, direction_bins, pos_data,
                            reldir_allframes
                        )
                    elif method == 2:
                        max_mrl, max_mrl_indices, mean_angle, mrl_values = find_consink_method2(
                            spike_train, reldir_occ_by_pos_cur, direction_bins, pos_data,
                            reldir_allframes, reldir_bin_idx
                        )
                    elif method == 3:
                        max_mrl, max_mrl_indices, mean_angle, mrl_values = find_consink_method3(
                            spike_train, reldir_occ_wholemaze, direction_bins, pos_data,
                            reldir_allframes
                        )
                    else:
                        raise ValueError("Method must be 1, 2 or 3")
                    
                    try:
                        consink_plat = max_mrl_indices[0][0] + 1
                    except:
                        breakpoint()
                    original_plat = np.where(platforms_trans == consink_plat)[0]

                    if len(original_plat) == 0:
                        original_plat = np.nan
                    else:
                        original_plat = original_plat[0] + 1

                    # store with goal suffix
                    consinks[unit_id][f'mrl_g{g}'] = max_mrl
                    consinks[unit_id][f'platform_g{g}'] = consink_plat
                    consinks[unit_id][f'original_platform_g{g}'] = original_plat
                    consinks[unit_id][f'mean_angle_g{g}'] = mean_angle
                    consinks[unit_id][f'numspikes_g{g}'] = len(spike_train)
                    consinks[unit_id][f'mrl_all_g{g}'] = mrl_values

            # Create dataframe
            consinks_df = pd.DataFrame(consinks).T
            print(consinks_df)

            # save as csv
            consinks_df.to_csv(os.path.join(output_folder, f'consinks_df_m{method}.csv'), index=False)
            print(f"Data saved to {os.path.join(output_folder, f'consinks_df_m{method}.csv')}")
            # save consinks_df
            save_pickle(consinks_df, f'consinks_df_m{method}', output_folder)

        # ######################### TEST STATISTICAL SIGNIFICANCE OF CONSINKS #########################
        # shift the head directions relative to their positions, and recalculate the tuning to the
        # previously identified consink position.

        if 1 in code_to_run:
            print("Assessing significance")
            # load the consinks_df
            consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)

            # make columns for the confidence intervals; place them directly beside the mrl column
            consinks_df = ensure_sig_columns(consinks_df, goals_to_include)

            for unit_id in tqdm(unit_ids):
                for g in goals_to_include:
                    if g == 0:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos_g0
                    elif g == 1:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                    elif g == 2:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
                    else:
                        reldir_occ_by_pos_cur = reldir_occ_by_pos

                    # print(f'Were at {unit_id} with { g}' )

                    

                    spike_train = get_spike_train(sorting, unit_id, pos_data, rawsession_folder, g=g, frame_rate=frame_rate,
                                                sample_rate=sample_rate)
                    if len(spike_train) < min_num_spikes:
                        continue
                    ci = recalculate_consink_to_all_candidates_from_translation(spike_train, pos_data,
                                                                                reldir_occ_by_pos_cur,
                                                                                direction_bins,
                                                                                reldir_allframes, reldir_occ_wholemaze,
                                                                                intervals_frames, reldir_bin_idx, method = method, goal=g)

                    consinks_df.loc[unit_id, f'ci_95_g{g}'] = ci[0]
                    consinks_df.loc[unit_id, f'ci_999_g{g}'] = ci[1]
                    mrl_val = consinks_df.loc[unit_id, f'mrl_g{g}']
                    if np.isfinite(ci[0]) and np.isfinite(mrl_val) and mrl_val > ci[0]:
                        sig = 'sig'
                    else:
                        sig = 'ns'
                    consinks_df.loc[unit_id, f'sig_g{g}'] = sig

            print(f"Saved consink data to the following folder: {output_folder}")
            try:
                consinks_df.to_csv(os.path.join(output_folder, f'consinks_df_m{method}.csv'))
            except:
                breakpoint()
            save_pickle(consinks_df, f'consinks_df_m{method}', output_folder)
        

        if 2 in code_to_run:
            print("Calculating mean sink position")
            hcoord, vcoord = get_coords_127sinks(derivatives_base)
            consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)
            average_sink = calculate_averagesink(consinks_df, hcoord, vcoord, goals_to_include)
            save_pickle(average_sink, f'average_sink_m{method}', output_folder)

        ######################## PLOT ALL CONSINKS #################################
        if 3 in code_to_run:
            hcoord, vcoord = get_coords_127sinks(derivatives_base)
            x_diff = np.mean(np.diff(hcoord))

            y_diff = np.mean(np.diff(vcoord))
            jitter = (2*x_diff, 2*y_diff)

            # Check if consinks_df is a dictionary otherwise convert
            consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)
            average_sink = load_pickle(f'average_sink_m{method}', output_folder)
            plot_all_consinks_127sinks(consinks_df, goal_numbers, hcoord, vcoord, platforms_trans,  jitter=jitter, plot_dir=output_folder,average_sink = average_sink, goals_to_include = goals_to_include,
                            plot_name=title, show_plots = show_plots)
            if len(methods)>1 and method != methods[-1]:
                plt.close('all')

    if 4 in code_to_run:
        # Plot fantail
        plot_fantail_mean_angles(derivatives_base, methods, output_folder, goals_to_include = goals_to_include, show_plots = show_plots)
        
    if 5 in code_to_run:
        pass
        # still to add
    export_sig_sinks(methods,output_folder,goals_to_include=goals_to_include)
def plot_fantail_mean_angles(derivatives_base, methods, output_folder, goals_to_include, show_plots = True,
                             n_bins=12):
    """
    Plot fantail (polar histograms) of mean angles for significant units,
    separately for each goal.
    """
    subject_id = derivatives_base.split(os.sep)[3]
    session_id = derivatives_base.split(os.sep)[4]
    title = f"Fantail {subject_id}, {session_id}, significant consinks"


    n_goals = len(goals_to_include)
    fig, axes = plt.subplots(
        len(methods), n_goals,
        figsize=(6 * n_goals, 4*len(methods)),
        subplot_kw={'projection': 'polar'}
    )
    plt.suptitle(title, size = 20)
    if n_goals == 1:
        axes = [axes]

    for i_m, method in enumerate(methods):
        consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)

        for i_g, g in enumerate(goals_to_include):
            ax = axes[i_m, i_g]

            sig_mask = consinks_df[f'sig_g{g}'] == 'sig'
            angles = consinks_df.loc[sig_mask, f'mean_angle_g{g}'].values
            if len(angles) == 0:
                ax.set_title(f'Method {method}, Goal {g}\nNo significant units')
                continue
            angles = angles[np.isfinite(angles)]

            if len(angles) == 0:
                ax.set_title(f'Method {method}, Goal {g}\nNo significant units')
                continue

            ax.hist(
                angles,
                bins=n_bins,
                range=(-np.pi, np.pi),
                density=True,
                alpha=0.6
            )

            mean_ang = circmean(angles)
            ax.plot(
                [mean_ang, mean_ang],
                [0, ax.get_rmax()],
                color='k',
                linewidth=2
            )

            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_title(
                f'Method {method}, Goal {g}\nN = {len(angles)}',
                fontsize=12
            )

    plt.tight_layout()
    save_path = os.path.join(
        output_folder, 'fantail_mean_angles_all_methods.png'
    )
    plt.savefig(save_path, dpi=300)
    if show_plots:
        plt.show()

    print(f"Saved fantail plot to {save_path}")

def plot_mrl_heatmaps_first_cells(
    derivatives_base,
    methods,
    output_folder,
    goal=1,
    n_cells=5,
    include_g0=False,
    cmap='viridis'
):
    """
    Plot MRL heatmaps over platform positions for the first n_cells.
    Rows = cells, Columns = methods
    """

    hcoord, vcoord = get_coords_127sinks(derivatives_base)
    platforms_trans = translate_positions()

    for j, method in enumerate(methods):
        consinks_df = load_pickle(f'consinks_df_m{method}', output_folder)

        for i, unit_id in enumerate(consinks_df.index[:n_cells]):
            ax = axes[i, j]

            mrl_all = consinks_df.loc[unit_id, f'mrl_all_g{goal}']
            if mrl_all is None:
                continue

            if not isinstance(mrl_all, np.ndarray):
                ax.set_title(f'Unit {unit_id}\nNo data')
                continue

            norm = Normalize(vmin=0, vmax=np.nanmax(mrl_all))
            cmap_fn = cm.get_cmap(cmap)

            
            # Add some coloured hexagons and adjust the orientation to match the rotated grid
            for s, (x, y) in enumerate(zip(hcoord, vcoord)):
                colour = cmap_fn(norm(mrl_all[s]))
                text = " "
                edgecolor = 'white'
                hex = RegularPolygon((x, y), numVertices=6, radius=83,
                                    orientation=np.radians(28),  # Rotate hexagons to align with grid
                                    facecolor=colour, alpha=0.2, edgecolor=edgecolor)
                ax.text(x, y, text, ha='center', va='center', size=15)  # Start numbering from 1
                ax.add_patch(hex)

            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')

            ax.set_title(
                f'Unit {unit_id}\nMethod {method}',
                fontsize=10)
    plt.suptitle(
        f'MRL heatmaps (goal {goal}) – first {n_cells} units',
        fontsize=18
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(
        output_folder,
        f'mrl_heatmaps_first_{n_cells}_cells_goal_{goal}.png'
    )
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved MRL heatmaps to {save_path}")
if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    goals_to_include = [1,2,3]
    main(derivatives_base, 'all trials', 'pyramidal', methods = [1, 2], code_to_run=[5])


