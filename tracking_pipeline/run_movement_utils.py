import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from movement.filtering import filter_by_confidence, interpolate_over_time
from movement.filtering import (
    rolling_filter,
)
import movement.kinematics as kin
import xarray as xr
from movement.kinematics import (
    compute_forward_vector_angle,
)
import ast
from movement.plots import plot_centroid_trajectory
import pandas as pd
from pathlib import Path
from typing import Any
from xarray.plot.facetgrid import FacetGrid
from movement.utils.vector import compute_norm
plt.ioff()


def create_folder(derivatives_base: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Create and return spatial behaviour data folders."""
    
    output_folder = derivatives_base / "analysis/spatial_behav_data"
    folder_directory = output_folder / "inference_results"
    positional_data_folder = output_folder / "XY_and_HD"
    movement_data_folder = output_folder / "movement_plots"
    trajectory_data_folder = output_folder / "animal_trajectory"

    for folder in (
        output_folder,
        folder_directory,
        positional_data_folder,
        movement_data_folder,
        trajectory_data_folder,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    return (
        output_folder,
        folder_directory,
        positional_data_folder,
        movement_data_folder,
        trajectory_data_folder,
    )

def plot_and_save_g(g: FacetGrid, movement_data_folder: Path, name: str, suptitle: str) -> None:
    """ Plots and saves g"""
    fig = g.fig
    output_path = movement_data_folder / name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.suptitle(suptitle)
    fig.show()

def get_xy(position: xr.DataArray, keypoints: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
    """ Obtains xy position of the keypoints and returns them"""
    if keypoints == 'ears':
        pos_xy = position.sel(keypoints=["left_ear", "right_ear"]).mean(dim="keypoints")
    else:
        pos_xy  = position.sel(keypoints=keypoints, drop=True)
    x_vals = pos_xy.sel(space = 'x')
    x =  x_vals.values.flatten()
    
    y_vals = pos_xy.sel(space = 'y')
    y =  y_vals.values.flatten()   
    return x, y 

def save_to_df(x: np.ndarray, y: np.ndarray, hd: np.ndarray, positional_data_folder: Path, name: str, speed: np.ndarray = None) -> None:
    """ Saves x, y, and hd to positional_data_folder under name.csv"""
    df = pd.DataFrame({
            "x":     x,
            "y":     y,
            "hd":    hd,
            "speed": speed if speed is not None else np.full_like(x, np.nan)
        })

    # save to CSV
    csv_path = positional_data_folder / name
    df.to_csv(csv_path, index=False)
    print(f"→ Saved positional + HD data to {csv_path}")
        
def ensure_da(arr: Any) -> xr.DataArray:
    """ Ensures arr is a dataarray"""
    return arr if hasattr(arr, "plot") else xr.DataArray(arr)

def get_position_values(ds: xr.DataArray, space: str) -> np.ndarray:
    # Returns positional values from ds
     return ds.sel(space=space).values.flatten()

def filter_ds_by_confidence(ds: xr.DataArray, threshold: float) -> tuple[xr.DataArray, FacetGrid]:
    """ Filters ds by confidence"""
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

def interpolate_ds_over_time(ds: xr.DataArray, max_gap: int)-> tuple[xr.DataArray, FacetGrid]:
    """ Fills in gaps in ds of max size max_gap"""
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
            
def smooth_ds(ds: xr.DataArray, window: int)-> xr.DataArray:
    """ smooths data set using window size window"""
    ds.update( 
        {
            "position": rolling_filter(
                ds.position, window, statistic="median", print_report=True
            )
        }
    )
    return ds
        
def calculate_hd(position: xr.DataArray) -> list:
    """ Calculating head direction using right angle between ears
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

def plot_trajectory(position: xr.DataArray, trajectory_data_folder: Path, tr: int) -> None:
    """ Plots trajectory of the rat and saves it"""
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
    output_path = trajectory_data_folder/f'trajectory_t{tr}.png'
    print(output_path)
    fig.savefig(output_path)
    fig.show()

def plot_speed(speed: np.ndarray, movement_data_folder: Path, tr: int, frame_rate: int, show_plots: bool, cutoff: float = 0.1) -> None:
    """ Plots speed of the rat and saves it"""

    cutoffs = [0.5, 1, 2, 5]
    
    fig, ax = plt.subplots(1, len(cutoffs), figsize=(10*len(cutoffs), 5))
    
    for cutoff, subplot in zip(cutoffs, ax):
        
        time = np.arange(len(speed)) / frame_rate

        # Mask values
        below = speed < cutoff
        above = speed >= cutoff

        # Plot above cutoff (default color)
        subplot.plot(time[above], speed[above], 'b. ', label="Speed ≥ cutoff")

        # Plot below cutoff in red
        subplot.plot(time[below], speed[below], 'r.', label=f"Speed < {cutoff} cm/s")

        subplot.set_xlabel('Time (s)')
        subplot.set_ylabel('Speed (cm/s)')
        subplot.set_title(f'Trial {tr}. Points below cutoff: {100*len(speed[below]) / len(speed):.1f}%')
        subplot.legend()


    plt.savefig(movement_data_folder / f't{tr}_speed.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

def find_filepath(folder_directory: Path, tr: int):
    """ Finds file path for h5 file for the trial"""
    matches = list(folder_directory.glob(f"*T{tr}_*.h5"))

    if len(matches) == 0:
        raise ValueError(f"Error: no h5 file found for trial {tr}. Quiting movement")
    else:
        file_path = matches[0]
    return file_path

def remove_unneeded_intervals(ds: xr.DataArray, tr: int, derivatives_base: Path, frame_rate: int = 25):
    """ Removes intervals we want to cut out from the positional data ds
    We obtain these from rawsession_folder/task_metadata/intervals_to_del"""
    rawsession_folder =  Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    df_path = rawsession_folder/ "task_metadata"/ "intervals_to_del.csv"
    
    # if path doesn't exists, we assume that there are no intervals to delete
    if not df_path.exists():
        return ds
    else:
        df = pd.read_csv(df_path)
        if pd.isna(df.iloc[tr-1, 1]):
            return ds # no need to remove any intervals
        interval = ast.literal_eval(df.iloc[tr-1, 1])
        if isinstance(interval[0],int): # if first element is float, we're dealing with only one interval to remove
            start = interval[0]
            end = interval[1]
            ds.position.data[start*frame_rate:end*frame_rate, :, :, :] = np.nan
        else:
            # if interval is an array with intervals
            intervals = interval
            for interval in intervals:
                start = interval[0]
                end = interval[1]
                ds.position.data[start*frame_rate:end*frame_rate, :, :, :] = np.nan
    return ds

def get_pixels_per_cm(derivatives_base: Path) -> float:
    """ Gets cm per pixel from maze_overlay_params.json"""
    params_path = derivatives_base / "analysis/maze_overlay/maze_overlay_params.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params["pixels_per_cm"]

def calculate_speed(x: np.ndarray, y: np.ndarray, pixels_per_cm: float, frame_rate: int = 25) -> np.ndarray:
    """ Calculates speed from position data"""
    # Calculate speed using the formula: speed = distance / time
    # Distance is calculated as the Euclidean distance between consecutive positions
    # Time is the inverse of the frame rate (time between frames)
    
    div_of_s =2 # how many halves of a second we look at. If 1, then look at 1/2 before, 1/2 after. If 2, then look at 1/4 before, 1/4 after. Etc.
    # For first 25 frames, calculate speed as distance between first frame and 25th frame
    val_int = int(frame_rate/(2*div_of_s))
    
    for i in range(len(x)):
        if i < frame_rate:
            dx = x[frame_rate] - x[0]
            dy = y[frame_rate] - y[0]
        elif i > len(x) - frame_rate:
            dx = x[-1] - x[-frame_rate-1]
            dy = y[-1] - y[-frame_rate-1]
        else:
            dx = x[i + val_int] - x[i - val_int]
            dy = y[i + val_int] - y[i - val_int]
            
        distance = np.sqrt(dx**2 + dy**2)
        distance_cm = distance / pixels_per_cm
        num_frames = 2 * val_int
        dt = num_frames / frame_rate
        speed_per_s = distance_cm / dt
        if not np.isfinite(speed_per_s):
            speed_per_s = np.nan
        if i == 0:
            speed = np.array([speed_per_s])
        else:
            speed = np.append(speed, speed_per_s)
    return speed

def compute_velocity_movement(position: xr.DataArray, pixels_per_cm: float, frame_rate: int = 25) -> np.ndarray:
    """ Computes velocity from position data
    Movement method, we're currently not using this"""
    velocity = kin.compute_velocity(position)
    speed = compute_norm(velocity)
    fig, ax = plt.subplots(5,1)
    keypoints = position.keypoints.values
    for i, keypoint in enumerate(keypoints):
        speed = compute_norm(velocity.sel(keypoints=keypoint))/ pixels_per_cm * frame_rate
        ax[i].plot(speed)
        ax[i].set_title(f"Velocity of {keypoint}")
    plt.show()
    #velocity = velocity.values / pixels_per_cm * frame_rate
    return velocity
