import numpy as np
import pandas as pd
import os
from utilities.create_intervals_specialbehav import create_intervals_specialbehav
from utilities.restrict_posdata_specialbehav import restrict_posdata_specialbehav
from utilities.create_sinks import create_sinks_and_dirbins
from find_popsink_newmethod import calculate_popsink

def check_restricted_df(rawsession_folder):
    """ Check if restricted_df.csv exists in the task_metadata folder"""
    task_metadata_folder = os.path.join(rawsession_folder, "task_metadata")
    restricted_df_path_csv = os.path.join(task_metadata_folder, "restricted_df.csv")
    restricted_df_path_xlsx = os.path.join(task_metadata_folder, "restricted_df.xlsx")
    if not os.path.exists(restricted_df_path_csv) and not os.path.exists(restricted_df_path_xlsx):
        raise FileNotFoundError(f"restricted_df.csv or restricted_df.xlsx not found in {task_metadata_folder}")

def check_trials_length(rawsession_folder):
    """ Check if trials_length.csv exists in the task_metadata folder"""
    task_metadata_folder = os.path.join(rawsession_folder, "task_metadata")
    trials_length_path = os.path.join(task_metadata_folder, "trials_length.csv")
    if not os.path.exists(trials_length_path):
        raise FileNotFoundError(f"trials_length.csv not found in {task_metadata_folder}")

def check_intervals_df(rawsession_folder):
    """ Check if restricted_final.csv exists in the task_metadata folder"""
    return os.path.exists(os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv"))

def prerequisites_popsink(derivatives_base, trials_to_include):
    """ Run this code before running popsink analysis
    Some prerequisites:
        1. restricted_df.csv must be saved to rawsession_folder/task_metadata
        2. 
    """
    run_zero = False
    folder_name = "popsink_data_newmethod"
    n_bins = 120 # number of sink bins
    
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # Check if restricted_df.csv exists
    check_restricted_df(rawsession_folder)
    check_trials_length(rawsession_folder)

   
    # Create restricted_final.csv
    create_intervals_specialbehav(rawsession_folder)
    
    # Creates XY_HD_goal{goal}_trials and XY_HD_allintervals.csv and shows the heatmap
    #restrict_posdata_specialbehav(derivatives_base, rawsession_folder,  frame_rate = 25) # NOTE: this one needs cleaning up
    
    # Create sink files and direction bins for hd
    # Also saves them to derivatives_base/analysis/spatial_features/{folder_name}/sink_bins.pkl
    create_sinks_and_dirbins(derivatives_base, n_bins = n_bins,folder_name = folder_name)
    
    calculate_popsink(derivatives_base, unit_type = 'pyramidal', code_to_run = [1, 2], title = 'Popsink Pyramidal', run_zero = run_zero, frame_rate = 25, sample_rate = 30000)
    
if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    prerequisites_popsink(derivatives_base, np.arange(1,14))