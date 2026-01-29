import numpy as np
from movement.io import load_poses
import matplotlib.pyplot as plt
from scipy.signal import welch
from movement import sample_data
from movement.filtering import filter_by_confidence, interpolate_over_time
from movement.kinematics import compute_velocity
from movement.filtering import (
    interpolate_over_time,
    rolling_filter,
    savgol_filter,
)
import xarray as xr
from movement.kinematics import (
    compute_forward_vector,
    compute_forward_vector_angle,
)
from movement.plots import plot_centroid_trajectory
from movement.utils.vector import cart2pol, pol2cart
import os
import glob
import pandas as pd
import json
import sys
plt.ioff()

def create_folder(derivatives_base):
    """ Creates folders that we need"""
    folder_directory = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'inference_results')
    output_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    positional_data_folder = os.path.join(output_folder, 'XY_and_HD')
    if not os.path.exists(positional_data_folder):
        os.makedirs(positional_data_folder)

    movement_data_folder = os.path.join(output_folder, 'movement_plots')
    if not os.path.exists(movement_data_folder):
        os.makedirs(movement_data_folder)

    trajectory_data_folder = os.path.join(output_folder, 'animal_trajectory')
    if not os.path.exists(trajectory_data_folder):
        os.makedirs(trajectory_data_folder)
        
    return output_folder, folder_directory, positional_data_folder, movement_data_folder, trajectory_data_folder

def plot_and_save_g(g, movement_data_folder, name, suptitle):
    fig = g.fig
    output_path = os.path.join(movement_data_folder , name)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.suptitle(suptitle)
    fig.show()

def get_xy(position, keypoints):
    if keypoints == 'ears':
        pos_xy = position.sel(keypoints=["left_ear", "right_ear"]).mean(dim="keypoints")
    else:
        pos_xy  = position.sel(keypoints=keypoints, drop=True)
    x_vals = pos_xy.sel(space = 'x')
    x =  x_vals.values.flatten()
    
    y_vals = pos_xy.sel(space = 'y')
    y =  y_vals.values.flatten()   
    return x, y 

def save_to_df(x, y, hd,positional_data_folder, name ):
    df = pd.DataFrame({
            "x":     x,
            "y":     y,
            "hd":    hd,
        })

    # save to CSV
    csv_path = os.path.join(positional_data_folder, name)
    df.to_csv(csv_path, index=False)
    print(f"→ Saved positional + HD data to {csv_path}")
        
def ensure_da(arr):
    return arr if hasattr(arr, "plot") else xr.DataArray(arr)

def get_position_values(ds, space):
    space_vals_arr = ds.sel(space=space)
    space_vals = space_vals_arr.values
    space_vals = space_vals.flatten()
    return space_vals

def filter_ds_by_confidence(ds, threshold):
    ds.update(
        {
            "position": filter_by_confidence(
                ds.position, ds.confidence, print_report=True, threshold = threshold
            )
        }
    )
    g = ds.position.squeeze().plot.line(
        x="time", row="keypoints", hue="space", aspect=2, size=2.5
    )
            
    return ds, g

def interpolate_ds_over_time(ds, max_gap):
    ds.update(
        {
            "position": interpolate_over_time(
                ds.position, max_gap=max_gap, print_report=True
            )
        }
    )
    g = ds.position.squeeze().plot.line(
        x="time", row="keypoints", hue="space", aspect=2, size=2.5
    )
            
    return ds, g
            
def smooth_ds(ds, window):
    ds.update(
        {
            "position": rolling_filter(
                ds.position, window, statistic="median", print_report=True
            )
        }
    )
    return ds
        
def calculate_hd(position):
    """ Calculating head direction in 3 different methods
    We do -el here, so that the head directions align with the videos
    """
    # Method 1: Orthogonal between left and right ear
    hd_orth_ears = compute_forward_vector_angle(
            position,
            left_keypoint="left_ear",
            right_keypoint="right_ear",
            reference_vector=(1, 0),  # positive x-axis
            camera_view="top_down",
            in_degrees=False,  # set to True for degrees
    )
    hd_orth_ears = hd_orth_ears.values.flatten()
    hd_orth_ears = [-el for el in hd_orth_ears]

    return hd_orth_ears

def plot_trajectory(position, trajectory_data_folder, tr):
    
    fig, ax = plt.subplots(1, 1)
    # Plot trajectories for each mouse on the same axes
    for rat_name, col in zip(
        position.individuals.values,
        ["r", "g", "b"],  # colours
        strict=False,
    ):
        plot_centroid_trajectory(
            position,
            individual=rat_name,
            ax=ax,  # Use the same axes for all plots
            c=col,
            marker="o",
            s=10,
            alpha=0.2,
        )
        ax.set_xlim(550,2050)
        ax.set_ylim(1750,100)
        ax.set_aspect('equal', adjustable='box') 
        ax.legend().set_alpha(1)
        ax.set_title(f'Trajectory trial {tr}')
    output_path = os.path.join(trajectory_data_folder, f'trajectory_t{tr}.png')
    print(output_path)
    fig.savefig(output_path)
    fig.show()
    
def find_filepath(folder_directory, tr):
    pattern = os.path.join(folder_directory, f"*T{tr}_*.h5")
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise ValueError(f"Error: no h5 file found for trial {tr}. Quiting movement")
    else:
        file_path = matches[0]
    return file_path

def run_movement(derivatives_base, trials_to_include, conf_threshold = 0.6, show_plots = False,  frame_rate = 25):
    """
    Processing the raw xy data as obtained from running inference.
    Creates:
    - XY_HD_t{tr}.csv files with xy position and hd
    - Plots of the position of keypoints before and after smoothing
    - Plots with animal trajectory (in animal trajectory folder)

    Inputs:
    derivatives_base: path to derivatives folder
    trials_to_include: trials to look at 
    show_plots: whether to show plots or just to save them
    frame_rate: frame rate of camera. Defaults to 25
    
    Steps:
    1. Filters by confidence
    2. Interpolates (creates plot)
    3. Smoothes (creates plot)
    4. Calculates HD (orthogonal to ears and other methods, though other methods are not used)
    5. Plots trajectory
    
    Notes:
    If camera will be changed, x and y lims for the animal trajecotry plots have to be changed
    Assumes that the inference data has the format "T{tr}_*.h5'
    """

    output_folder, folder_directory, positional_data_folder, movement_data_folder, trajectory_data_folder = create_folder(derivatives_base)
    

    for tr in trials_to_include:
        file_path = find_filepath(folder_directory, tr)

        # --------- Plotting confidence and position --------
        ds = load_poses.from_sleap_file(file_path, fps=frame_rate)
        position = ds.position

        g = ds.confidence.squeeze().plot.line(
            x="time", row="keypoints", aspect=2, size=2.5
        )
        plot_and_save_g(g, movement_data_folder,f"t{tr}_confidence_raw.png", "Raw Confidence")

        g = ds.position.squeeze().plot.line(
            x="time", row="keypoints", hue="space", aspect=2, size=2.5
        )
        plot_and_save_g(g, movement_data_folder, f"t{tr}_keypoints_raw.png", "Raw Position")


        # ---------- Filtering by confidence ----------
        ds, g = filter_ds_by_confidence(ds, threshold = conf_threshold)

        plot_and_save_g(g, movement_data_folder, f"t{tr}_filtered_by_confidence.png", "Raw Position")


        # ---------- Interpolating over time ----------
        max_gap = 30
        ds, g  = interpolate_ds_over_time(ds, max_gap = max_gap)
        
        plot_and_save_g(g, movement_data_folder, f"t{tr}_keypoints_interpolated.png", "With interpolation, max_gap = {max_gap}")
        

        # ---------- Smoothing -----------------------
        window = int(ds.fps * 0.5)
        ds_smooth = smooth_ds(ds, window)

        g = ds_smooth.position.squeeze().plot.line(
            x="time", row="keypoints", hue="space", aspect=2, size=2.5
        )
        plot_and_save_g(g, movement_data_folder,  f"t{tr}_keypoints_smoothed.png", f"Smoothed, window = {window}")


        #------------- HEAD DIRECTION ----------------
        position = ds_smooth.position
        hd_orth_ears = calculate_hd(position)
        
        # Getting midpoint between left and right ear
        x, y = get_xy(position, 'ears')
    
        # HD that we'll be using: orthogonal to xy
        hd  = hd_orth_ears.values if hasattr(hd_orth_ears, "values") else hd_orth_ears

        # Save x, y, and hd to df
        save_to_df(x, y, hd, positional_data_folder,  f"XY_HD_t{tr}.csv")


        # ------------- Plotting trajectory------------- 
        plot_trajectory(position, trajectory_data_folder, tr)

        
        # Saving center coordinates (used in consink code)
        x, y = get_xy(position, 'center')
        save_to_df(x, y, hd, positional_data_folder,  f"XY_HD_center_t{tr}.csv")
        
        if show_plots:
            print(f"Showing plots for trial {tr}. Close all to continue")
            plt.show(block = True)
        if not show_plots:
            plt.close('all')
            
         # -------------Skeleton data------------- 
        
        # Saving center coordinates (used in consink code)
        x_center, y_center = get_xy(position, 'center')
        x_le, y_le = get_xy(position, 'left_ear')
        x_re, y_re = get_xy(position, 'right_ear')
        
        df = pd.DataFrame({
            "x_center":     x_center,
            "y_center":     y_center,
            "x_le":     x_le,
            "y_le":     y_le,
            "x_re":     x_re,
            "y_re":     y_re,
            "hd":    hd,
        })

        # save to CSV
        csv_path = os.path.join(positional_data_folder, f"XY_HD_skeleton_t{tr}.csv")
        df.to_csv(csv_path, index=False)
        print(f"→ Saved positional + HD data to {csv_path}")
        


def main():
    data = json.loads(sys.stdin.read())
    
    run_movement(
        data["derivatives_base"],
        data["trials_to_include"]
    )

    print(json.dumps({"status": "done"}))


if __name__ == "__main__":
    derivatives_base =  r"C:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    
    run_movement(derivatives_base, [2], show_plots = True)