import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from calculate_occupancy import get_relative_direction_occupancy_by_position,  get_axes_limits,  get_direction_bins
from find_consinks_main_functions import find_consink, recalculate_consink_to_all_candidates_from_translation, find_consink_method2, find_consink_method3
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from utilities.trials_utils import get_limits_from_json, get_goal_coordinates, get_coords
from matplotlib.patches import RegularPolygon
import matplotlib
matplotlib.use("QtAgg") 
from tqdm import tqdm
from typing import Literal
cm_per_pixel = 1


    
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

    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_goal0_trials.csv')
    pos_data_g0 = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data_g0['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data_g0['hd'] = np.deg2rad(pos_data_g0['hd'])

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
    

    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics',  'spatial_features', 'consink_data_newmethod')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Direction bins (from -pi to pi)
    direction_bins = get_direction_bins(n_bins=12)

    # Loading or creating data 
    file_name = 'reldir_occ_by_pos.npy'
    
    if not os.path.exists(os.path.join(output_folder, file_name)) or not os.path.exists(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy')):
        print("Calculating relative direction occupancy by position")
        reldir_occ_by_pos, sink_bins, candidate_sinks = get_relative_direction_occupancy_by_position(pos_data_reldir, limits)
        reldir_occ_by_pos_g0, _, _ = get_relative_direction_occupancy_by_position(pos_data_g0, limits)
        reldir_occ_by_pos_g1, _, _ = get_relative_direction_occupancy_by_position(pos_data_g1, limits)
        reldir_occ_by_pos_g2, _, _ = get_relative_direction_occupancy_by_position(pos_data_g2, limits)
        np.save(os.path.join(output_folder , file_name), reldir_occ_by_pos)
        np.save(os.path.join(output_folder , 'reldir_occ_by_pos_g0.npy'), reldir_occ_by_pos_g0)
        np.save(os.path.join(output_folder , 'reldir_occ_by_pos_g1.npy'), reldir_occ_by_pos_g1)
        np.save(os.path.join(output_folder , 'reldir_occ_by_pos_g2.npy'), reldir_occ_by_pos_g2)
        
        # save sink bins and candidate sinks as pickle files
        save_pickle(sink_bins, 'sink_bins', output_folder)
        save_pickle(candidate_sinks, 'candidate_sinks', output_folder)
        save_pickle(direction_bins, 'direction_bins', output_folder)     

    else:
        print("Loading reldir occ, not callculating")
        reldir_occ_by_pos = np.load(os.path.join(output_folder, file_name))
        reldir_occ_by_pos_g0 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g0.npy'))
        reldir_occ_by_pos_g1 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy'))
        reldir_occ_by_pos_g2 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g2.npy'))
        sink_bins = load_pickle('sink_bins', output_folder)
        candidate_sinks = load_pickle('candidate_sinks', output_folder)

    ## Get goal coordinates
    # Doesn't do antyhing yet
    goal_coordinates = get_goal_coordinates(derivatives_base, rawsession_folder)

    ################# CALCULATE CONSINKS ###########################################     
    consinks = {}
    consinks_df = {}

    if 0 in code_to_run:
        print("Calculating consinks")
        for unit_id in tqdm(unit_ids):
            consinks[unit_id] = {'unit_id': unit_id}

            for g in [0, 1, 2]:
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
                    spike_train, reldir_occ_by_pos_cur, sink_bins, direction_bins, candidate_sinks, pos_data
                )
                consink_position = np.round(
                    [candidate_sinks['x'][max_mrl_indices[1][0]], 
                    candidate_sinks['y'][max_mrl_indices[0][0]]], 3
                )

                # store with goal suffix
                consinks[unit_id][f'mrl_g{g}'] = max_mrl
                consinks[unit_id][f'position_g{g}'] = consink_position
                consinks[unit_id][f'mean_angle_g{g}'] = mean_angle

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
            idx_g1 = consinks_df.columns.get_loc('mrl_g2')
            consinks_df.insert(idx_g1 + 1, 'ci_95_g1', np.nan)
            consinks_df.insert(idx_g1 + 2, 'ci_999_g1', np.nan)
            idx_g2 = consinks_df.columns.get_loc('mrl_g2')
            consinks_df.insert(idx_g2 + 1, 'ci_95_g2', np.nan)
            consinks_df.insert(idx_g2 + 2, 'ci_999_g2', np.nan)

        for unit_id in tqdm(unit_ids):
            for g in [0, 1,2]:
                reldir_occ_by_pos_cur = f'reldir_occ_by_pos_g{g}' # Not splitting by position
                
                spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
                spike_train_secs = spike_train_unscaled / sample_rate
                spike_train_secs_g = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder, goal=g)
                spike_train = np.round(spike_train_secs_g*frame_rate) # trial data is now in frames in order to match it with xy data
                spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]  # Ensure spike train is within bounds of x and y

                if len(spike_train) == 0:
                    continue
                ci = recalculate_consink_to_all_candidates_from_translation(spike_train, pos_data, reldir_occ_by_pos_cur, sink_bins, direction_bins, candidate_sinks)

                consinks_df.loc[unit_id, f'ci_95_g{g}'] = ci[0]
                consinks_df.loc[unit_id, f'ci_999_g{g}'] = ci[1]
        print(f"Saved consink data to the following folder: {output_folder}")
        consinks_df.to_csv(os.path.join(output_folder, 'consinks_df.csv'))
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
    # Check if consinks_df is a dictionairy otherwise convert
    consinks_df = load_pickle('consinks_df', output_folder)
    

    plot_all_consinks(consinks_df, goal_coordinates, hcoord, vcoord, limits, jitter=jitter, plot_dir=plot_dir, plot_name='ConSinks Good Units')

if __name__ == "__main__":
    
    
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    main(derivatives_base, 'intervals', 'good', code_to_run=[0, 1,2])


