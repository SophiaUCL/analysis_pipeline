import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import spikeinterface.extractors as se
import pandas as pd
from typing import Literal
from HCT_analysis.utilities.trials_utils import get_direction_bins
from HCT_analysis.utilities.load_and_save_data import load_pickle, save_pickle
from HCT_analysis.utilities.trials_utils import  get_goal_numbers, get_pos_data, get_spike_train, get_sink_positions_platforms, translate_positions
from HCT_analysis.consinks.find_consinks_main_functions import get_reldir_bin_idx, find_consink, get_reldir_occ_wholemaze, recalculate_consink_to_all_candidates_from_translation, find_consink_method2, find_consink_method3, get_dir_allframes, calculate_reldir_by_pos


RelDirOccTypes= Literal['all trials', 'intervals']
num_candidate_sinks = 127


def calculate_popsink_allspikes(derivatives_base: Path, load_units_which_method: int, rel_dir_occ: RelDirOccTypes, goals_to_include: list = [0,1,2], methods: list = [1,2,3], code_to_run: list = [0,1,2], frame_rate: int = 25, sample_rate: int = 30000):
    
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Loading xy data
    pos_data, pos_data_g0, pos_data_g1, pos_data_g2, pos_data_reldir = get_pos_data(derivatives_base, rel_dir_occ, goals_to_include)
    
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting", "sorter_output")
    sorting = se.read_kilosort(
        folder_path=kilosort_output_path
    )
    
    # Get all the pyramidal units
    consink_folder = derivatives_base/'analysis'/'cell_characteristics'/'spatial_features'/'consinks'
    consink_units_path = consink_folder/f'significant_consink_unit_ids_method_{load_units_which_method}.npy'
    unit_ids = np.load(consink_units_path, allow_pickle = True)
    unit_item = unit_ids.item()
    unit_ids = unit_item[1]
    
    # Output folder
    output_folder = derivatives_base/'analysis'/'cell_characteristics'/'spatial_features'/'popsinks_allspikes'
    output_folder.mkdir(parents=True, exist_ok=True)
    # Direction bins (from -pi to pi)
    direction_bins = get_direction_bins(n_bins=12)
    
    # Intervals frames
    path = os.path.join(rawsession_folder, 'task_metadata', 'restricted_df_frames.csv')
    intervals_frames = pd.read_csv(path)


   # gets translated positions
    platforms_trans = translate_positions()
    # Loading or creating data
    sink_positions = get_sink_positions_platforms(derivatives_base)
    goal_numbers= get_goal_numbers(derivatives_base)
    _, reldir_allframes = get_dir_allframes(pos_data, sink_positions)
    reldir_occ_wholemaze = get_reldir_occ_wholemaze(reldir_allframes, direction_bins)
    reldir_bin_idx = get_reldir_bin_idx(reldir_allframes, direction_bins)
    file_name = 'reldir_occ_by_pos.npy'
    
    # Loading xy data
    pos_data, pos_data_g0, pos_data_g1, pos_data_g2, pos_data_reldir = get_pos_data(derivatives_base, rel_dir_occ, goals_to_include)
    
    if -1 in code_to_run:
        print("Calculating relative direction occupancy by position")
        calculate_reldir_by_pos(output_folder, sink_positions, pos_data_reldir,pos_data_g0,pos_data_g1,pos_data_g2,goals_to_include)

   
    reldir_occ_by_pos = np.load(os.path.join(output_folder, file_name))
    if 0 in goals_to_include:
        reldir_occ_by_pos_g0= np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g0.npy'))
    if 1 in goals_to_include:
        reldir_occ_by_pos_g1 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g1.npy'))
    if 2 in goals_to_include:
        reldir_occ_by_pos_g2 = np.load(os.path.join(output_folder, 'reldir_occ_by_pos_g2.npy'))

    ################# CALCULATE CONSINKS ###########################################
    
    results = []
    
    for unit_method in methods:
        print(f"Loading consinks from method {unit_method}")
        consink_units_path = consink_folder/f'significant_consink_unit_ids_method_{unit_method}.npy'
        unit_ids = np.load(consink_units_path, allow_pickle = True)
        unit_item = unit_ids.item()
        unit_ids = unit_item[1]
        
        for method in methods:
            print(f"Method {method}")

            population_spikes_all_goals= {}
            

            print("Calculating consinks")
            for g in goals_to_include: 
                if g == 0:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g0
                elif g == 1:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g1
                elif g == 2:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos_g2
                else:
                    reldir_occ_by_pos_cur = reldir_occ_by_pos
            spike_train_allunits = []
            for unit_id in tqdm(unit_ids):
                spike_train = get_spike_train(sorting, unit_id, pos_data, rawsession_folder, g=g, frame_rate=frame_rate,
                                            sample_rate=sample_rate)

                spike_train_allunits = np.append(spike_train_allunits, spike_train)
            
            spike_train_allunits = spike_train_allunits.astype(int)
            population_spikes_all_goals[g] = spike_train_allunits
            
            # get consink
            if method == 1:
                max_mrl, max_mrl_indices, mean_angle, mrl_values = find_consink(
                    spike_train_allunits, reldir_occ_by_pos_cur, direction_bins, pos_data,
                    reldir_allframes
                )
            elif method == 2:
                max_mrl, max_mrl_indices, mean_angle, mrl_values = find_consink_method2(
                    spike_train_allunits, reldir_occ_by_pos_cur, direction_bins, pos_data,
                    reldir_allframes, reldir_bin_idx
                )
            elif method == 3:
                max_mrl, max_mrl_indices, mean_angle, mrl_values = find_consink_method3(
                    spike_train_allunits, reldir_occ_wholemaze, direction_bins, pos_data,
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

            print("Assessing significance")
            ci = recalculate_consink_to_all_candidates_from_translation(spike_train, pos_data,
                                                        reldir_occ_by_pos_cur,
                                                        direction_bins,
                                                        reldir_allframes, reldir_occ_wholemaze,
                                                        intervals_frames, reldir_bin_idx, method = method, goal=g)
            
            results.append({
                "unit_method": unit_method,
                "method": method,
                "goal": g,
                "mrl": float(max_mrl),
                "platform": int(consink_plat),
                "original_platform": original_plat,
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "mean_angle": float(mean_angle),
                'sig': bool(ci[0] < max_mrl)
            })
        save_pickle(spike_train_allunits,f'spike_train_allunits_consinks_method_{unit_method}.pkl',  output_folder)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_folder/'popsink_results_allspikes.csv', index=False)








if __name__ == "__main__":
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
    load_units_which_method = 1
    rel_dir_occ = "all trials"
    goals_to_include = [1]
    methods = [1,2,3]
    code_to_run = [-1, 0,1,2]
    
    calculate_popsink_allspikes(Path(derivatives_base), load_units_which_method, rel_dir_occ, goals_to_include, methods, code_to_run)
                
