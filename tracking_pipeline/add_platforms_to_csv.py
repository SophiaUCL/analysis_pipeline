import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from matplotlib.path import Path

def add_platforms_to_all(pos_data, params):
    platforms = []
    hcoord = params["hcoord_tr"]
    vcoord = params["vcoord_tr"]
    hex_side_length = params["hex_side_length"]
    
    print("Adding platforms to positional data")
    for i in tqdm(range(len(pos_data))):
        x = pos_data['x'].iloc[i]
        y = pos_data['y'].iloc[i]
        if np.isnan(x):
            plat = np.nan
        else:
            plat = get_nearest_platform(x, y, hcoord, vcoord, hex_side_length)
        platforms.append(plat)
        
    pos_data['platform'] = platforms
    return pos_data

def get_nearest_platform(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):
    platform = get_platform_number(rat_locx, rat_locy, hcoord, vcoord, hex_side_length)
    if not np.isnan(platform):
        return np.int32(platform)
    else:
        min_dist = 200
        closest_platform = np.nan
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            dist = np.sqrt((rat_locx - x)**2 + (rat_locy - y)**2)

            if dist < min_dist:
                closest_platform = i + 1
                min_dist = dist
        return closest_platform

def is_point_in_platform(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):
    hex_vertices = []
    for angle in np.linspace(0, 2 * np.pi, num=6, endpoint=False):
        hex_vertices.append([
            hcoord + hex_side_length * np.cos(angle),
            vcoord + hex_side_length * np.sin(angle)
        ])
    hexagon_path = Path(hex_vertices)
    return hexagon_path.contains_point((rat_locx, rat_locy))

# Updated code to find the hexagon the rat is in
def get_platform_number(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if is_point_in_platform(rat_locx, rat_locy, x, y, hex_side_length):
            return i + 1
    return np.nan

def add_platforms_to_csv(derivatives_base, file = 'center', frame_rate = 25):
    """
    Adds platforms to XY_HD_alltrials_center.csv and saves it as XY_HD_w_platforms.csv

    Input:
    dervitives_base: path to derivatives folder
    
    """
    if file is not 'center' and file is not 'intervals':
        raise ValueError('File should be either center or intervals')

    name = 'XY_HD_alltrials_center.csv' if file is 'center' else 'XY_HD_allintervals.csv'
    # Loading positional data
    folder_path = os.path.join(derivatives_base, "analysis", "spatial_behav_data", "XY_and_HD")
    pos_path = os.path.join(folder_path, name)
    
    if not os.path.exists(pos_path):
        print("Error, XY_HD_alltrials_center.csv not found")
        
    pos_data = pd.read_csv(pos_path)
    
    # Loading hexagon parameters
    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    with open(params_path, "r") as f:
        params = json.load(f)

    pos_data_w_plat = add_platforms_to_all(pos_data, params)
    output_name = 'XY_HD_w_platforms.csv' if file is 'center' else 'XY_HD_allintervals_w_platforms.csv'
    output_path = os.path.join(folder_path, output_name)
    pos_data_w_plat.to_csv(output_path, index = False)
    print(f"Saved data to {output_path}")

    platforms = pos_data_w_plat['platform']
    x = pos_data_w_plat['x']

    platforms_inval = np.isnan(platforms)
    x_inval = np.isnan(x)

    platforms = platforms[~platforms_inval]
    x = x[~x_inval]

    num_missed = len(x) - len(platforms)
    num_missed_s = num_missed/frame_rate
    print(f"Missing {num_missed_s} seconds of platform data (compared to x data)")
    
    

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    add_platforms_to_csv(derivatives_base)
    
    
    