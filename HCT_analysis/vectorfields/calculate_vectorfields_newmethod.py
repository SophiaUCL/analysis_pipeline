import numpy as np
import matplotlib
matplotlib.use('Agg')
import spikeinterface.extractors as se
import pandas as pd
from pathlib import Path
from typing import Literal
from HCT_analysis.utilities.load_and_save_data import load_pickle, save_pickle
from HCT_analysis.utilities.trials_utils import get_goal_coordinates, get_unit_ids
from HCT_analysis.vectorfields.vectorfields_utils import plot_vector_fields_all, calculate_vector_fields

UnitTypes = Literal['pyramidal', 'good', 'all', 'test']

def main(derivatives_base: Path,  unit_type: UnitTypes, methods: list = [1,2,3], goals_to_include: list = [0,1,2]) -> None:
    """
    For each unit, creates a 3x1 subplot of vector fields for all trials, goal 1 trials, and goal 2 trials.
    run get_directional_occupancy_by_pos.py first to generate the necessary data.

    Inputs
    --------
    derivatives_base (Path): The base directory for the derivatives.
    unit_type (pyramidal, good, all, or test): Which units to include in the analysis
    methods (list): What methods to load. Currently not in use
    goals_to_include (list): Restrict position and data to the goal.
        
    Outputs
    --------
    To: derivatives_base / 'analysis' / 'cell_characteristics' /  'spatial_features' / 'consinks'
    It saves  'vector_fields.pkl' and 'mean_resultant_lengths.pkl', with vectorfields and MRLs for each goal
    
    To: derivatives_base / 'analysis' / 'cell_characteristics' / 'spatial_features' / 'vector_fields'
    It saves pngs of the vector fields for each unit, with the consinks overlaid. One plot per unit.
    
    Called by
    -------
    HCT_pipeline.py
    """
    
    rawsession_folder= Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Loading spike data
    kilosort_output_path = derivatives_base / "ephys" / "concat_run" / "sorting" / "sorter_output"
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    unit_ids = get_unit_ids(derivatives_base, unit_ids, unit_type)
    
    # Loading xy data
    pos_data_path = derivatives_base / 'analysis' / 'spatial_behav_data' / 'XY_and_HD' / 'XY_HD_alltrials.csv'
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])


    output_folder = derivatives_base / 'analysis' / 'cell_characteristics' /  'spatial_features' / 'consinks'
    output_folder.mkdir(parents=True, exist_ok=True)


    goal_coordinates = get_goal_coordinates(derivatives_base, rawsession_folder)
    
    vector_fields = {}
    mean_resultant_lengths = {}
        

    ############# CALCULATING VECTOR FIELDS ##############3
    

    # Now per goal
    for g in goals_to_include:# AGAIN, We're not doing full trial here
        print("Calculating vector fields for goal ", g)
        spike_rates_by_position_and_direction_by_goal = load_pickle( f'spike_rates_by_position_and_direction_g{g}', output_folder)
        vector_fields[g], mean_resultant_lengths[g] = calculate_vector_fields(spike_rates_by_position_and_direction_by_goal)
    print("Saving")
    
    # Save
    save_pickle(vector_fields, 'vector_fields', output_folder)
    save_pickle(mean_resultant_lengths, 'mean_resultant_lengths', output_folder)
    
    # Plotting
    plot_dir = derivatives_base / 'analysis' / 'cell_characteristics' / 'spatial_features' / 'vector_fields'
    plot_dir.mkdir(parents = True, exist_ok = True)
    print(f"Saving to {plot_dir}")
    
    # Plotting
    x_bins = vector_fields[g]['x_bins']
    x_centres = x_bins[:-1] + np.diff(x_bins)/2
    y_bins = vector_fields[g]['y_bins']
    y_centres = y_bins[:-1] + np.diff(y_bins)/2
    
    plot_vector_fields_all(derivatives_base, unit_ids,vector_fields,goal_coordinates,x_centres,y_centres,plot_dir,goals_to_include,methods,output_folder)



if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    main(derivatives_base, rawsession_folder)










    