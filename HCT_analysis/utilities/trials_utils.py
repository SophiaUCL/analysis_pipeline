
import os
import glob 
import pandas as pd
import numpy as np
import json
from HCT_analysis.utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav

def ensure_sig_columns(consinks_df, goals):
    for g in goals:
        for col in [f'ci_95_g{g}', f'ci_999_g{g}', f'sig_g{g}']:
            if col not in consinks_df.columns:
                consinks_df[col] = np.nan
    return consinks_df

def append_alltrials(derivatives_base):
    """ 
    Takes the alltrials csv and creates a new csv only with the rows of the trial date

    Args:
        rawsession_folder (str): path to the rawdata folder
    
    Exports:
       rawsession/behaviour/alltrials_trialday.csvL alltrials filtered for only this day
       
    NOTE: this will most likely still include trials that are not the trials we will use. These have to be removed later. 
    """
    rawsession_folder =  derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    folder = os.path.basename(rawsession_folder)        # Obtains session name, example: "ses-02_date-05092025"
    date = folder.split("date-")[-1]  # Obtains number after 'date', for example "05092025"
    
    if os.path.exists(os.path.join(rawsession_folder,"behaviour", "alltrials_trialday.csv")):
        print("Trialday csv exists, exiting")
        return
    alltrials_paths = glob.glob(os.path.join(rawsession_folder,"behaviour" ,"alltrials*.csv"))
    alltrials_path = alltrials_paths[0]
    
    df = pd.read_csv(alltrials_path)
    
    # If date starts with 0, remove it (its saved without 0 in the trial_csv)
    if date.startswith("0"):
        date = date[1:]

    df = df[df['Date'] == int(date)]
    
    output_path = os.path.join(rawsession_folder,"behaviour", "alltrials_trialday.csv")
    df.to_csv(output_path, index=False)
    print(f"Created {output_path}")

def get_pos_data(derivatives_base, rel_dir_occ):
    """ Get positional data"""
    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                 'XY_HD_w_platforms.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])

    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                 'XY_HD_goal0_trials.csv')
    try:
        pos_data_g0 = pd.read_csv(pos_data_path)

        if np.nanmax(pos_data_g0['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
            pos_data_g0['hd'] = np.deg2rad(pos_data_g0['hd'])
    except:
        pos_data_g0 = None
        print("No data found for g0. Returning None")

    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                 'XY_HD_goal1_trials.csv')
    pos_data_g1 = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data_g1['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
        pos_data_g1['hd'] = np.deg2rad(pos_data_g1['hd'])

    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                 'XY_HD_goal2_trials.csv')
    pos_data_g2 = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data_g2['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
        pos_data_g2['hd'] = np.deg2rad(pos_data_g2['hd'])

    # Getting the positional data we'll use for the rel_dir_occ
    if rel_dir_occ == 'all trials':
        name = 'XY_HD_w_platforms.csv'
    elif rel_dir_occ == 'intervals':
        name = 'XY_HD_allintervals_w_platforms.csv'
    pos_data_reldir_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', name)
    pos_data_reldir = pd.read_csv(pos_data_reldir_path)

    if np.nanmax(pos_data_reldir['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
        pos_data_reldir['hd'] = np.deg2rad(pos_data_reldir['hd'])
    return pos_data, pos_data_g0, pos_data_g1, pos_data_g2, pos_data_reldir


def get_spike_train(sorting, unit_id, pos_data, rawsession_folder, g, frame_rate=25, sample_rate=30000):
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

def get_sink_positions_platforms(derivatives_base):
    """ Gets sink position for the 127 platform sinks"""
    path = os.path.join(derivatives_base, 'analysis', 'maze_overlay', 'maze_overlay_params_consinks.json')
    with open(path) as f:
        data = json.load(f)
    hcoord = data['hcoord_tr']
    vcoord = data['vcoord_tr']
    sink_positions = [[hcoord[s], vcoord[s]] for s in range(len(hcoord))]
    return sink_positions

def translate_positions():
    """ We use an overlay of 127 consink positions, whilst there's 61 platforms. So platform 1-61 doesn't map to the consink position. This is an array that gives the rigth platform numbers"""

    arr1 = np.arange(18,23).tolist() #corresponds to platform 1 -5
    arr2 = np.arange(27,33).tolist() # 6-11, etc
    arr3 = np.arange(37,44).tolist()
    arr4 = np.arange(48,56).tolist()
    arr5 = np.arange(60,69).tolist()
    arr6 = np.arange(73,81).tolist()
    arr7 = np.arange(85,92).tolist()
    arr8 = np.arange(96,102).tolist()
    arr9 = np.arange(106,111).tolist()
    platforms_trans = arr1 + arr2 + arr3 + arr4 + arr5 + arr6 + arr7 + arr8 + arr9
    platforms_trans = np.array(platforms_trans)

    if len(platforms_trans) !=  61:
        print(f"length platforms_trans = {len(platforms_trans)}")
        raise ValueError("Platforms trans should have length 61 ")

    return platforms_trans
def verify_allnans(spike_train, pos_data):
    """ Verifies that not all values are nan values"""
    x_org = pos_data.iloc[:, 0].to_numpy()
    hd_org = pos_data.iloc[:, 2].to_numpy()

    # spike_train = [el for el in spike_train if el < len(x_org)]  # Ensure spike train is within bounds of x and y
    # Finding spike times for this unit
    x = x_org[spike_train]
    hd = hd_org[spike_train]

    mask = np.isnan(x) | np.isnan(hd)
    false_vals = np.where(mask == False)[0]
    if len(false_vals) < 2:
        return True
    else:
        return False

def get_goal_numbers(derivatives_base):
    """
    Obtains goal numbers from alltrials_trialday.csv
    
    Args:
        rawsession_folder (str): path to the rawdata folder
    
    Returns:
        [goal1, goal2]
    """
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    df_path = os.path.join(rawsession_folder, "behaviour", "alltrials_trialday.csv")
    df = pd.read_csv(df_path)
    goal1 = df['Goal 1'].values[0]
    goal2 = df['Goal 2'].values[0]
    
        
    return [np.int32(goal1), np.int32(goal2)]

def get_coords(derivatives_base):
    params_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    with open(params_path) as f:
        params= json.load(f)
    hcoord_tr = params["hcoord_tr"]
    vcoord_tr= params["vcoord_tr"]
    return hcoord_tr, vcoord_tr

def get_coords_127sinks(derivatives_base):
    params_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params_consinks.json")
    with open(params_path) as f:
        params= json.load(f)
    hcoord_tr = params["hcoord_tr"]
    vcoord_tr= params["vcoord_tr"]
    return hcoord_tr, vcoord_tr

def get_goal_coordinates(derivatives_base, rawsession_folder):
    """
    Returns:
        Goal coordinates. If json file with them doesn't exist, it makes it
    """
    coords_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "goal_coords.json")
    if not os.path.exists(coords_path):
            ## Gets goal coordinates
        goals = get_goal_numbers(derivatives_base)
        goal1 = goals[0]
        goal2 = goals[1]
        params_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
        with open(params_path) as f:
            params= json.load(f)
        hcoord_tr = params["hcoord_tr"]
        vcoord_tr= params["vcoord_tr"]
        
        goal1_coords = [hcoord_tr[np.int32(goal1 -1)], vcoord_tr[np.int32(goal1 - 1)]]
        goal2_coords = [hcoord_tr[np.int32(goal2 -1)], vcoord_tr[np.int32(goal2 - 1)]]
        
        coords_path =  os.path.join(derivatives_base, "analysis", "maze_overlay", "goal_coords.json")
        coords = {
            "goal1_coords": goal1_coords,
            "goal2_coords": goal2_coords
        }
        os.makedirs(os.path.dirname(coords_path), exist_ok=True)
        with open(coords_path, 'w') as f:
            json.dump(coords, f, indent=4)
        
    with open(coords_path) as f:
        data= json.load(f)

    goal1_coords = data["goal1_coords"]
    goal2_coords = data["goal2_coords"]
    return [goal1_coords, goal2_coords]

def get_limits_from_json(derivatives_base):
    """Gets the xy limits from the json file created in the get_limits.py function"""
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    try:
        x_min = limits["x_min"]
        y_min = limits["y_min"]
        x_max = limits["x_max"]
        y_max = limits["y_max"]
    except:
        x_min = limits["xmin"]
        y_min = limits["ymin"]
        x_max = limits["xmax"]
        y_max = limits["ymax"]

    return x_min, x_max, y_min, y_max

def get_unit_ids(derivatives_base, unit_ids, unit_type):
    # Getting unit IDs depending on type
    if unit_type == 'good':
        good_units_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting", "sorter_output",
                                       "cluster_group.tsv")
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
        # Loading pyramidal units
    elif unit_type == 'pyramidal':
        pyramidal_units_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features",
                                            "all_units_overview", "pyramidal_units_2D.csv")
        print("Getting pyramidal units 2D")
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
    elif unit_type == "test":
        print("getting first 5 units")
        unit_ids = unit_ids[:5]
    return unit_ids

if __name__ == "__main__":
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-02_date-05092025"
    append_alltrials(rawsession_folder)
    goal_numbers = get_goal_numbers(rawsession_folder)
    print(f"Goal numbers: {goal_numbers}")