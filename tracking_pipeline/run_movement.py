from movement.io import load_poses
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from tracking_pipeline.run_movement_utils import compute_velocity_movement, plot_speed, calculate_speed, get_pixels_per_cm, find_filepath, get_xy, save_to_df, interpolate_ds_over_time, remove_unneeded_intervals, filter_ds_by_confidence, plot_and_save_g, create_folder, plot_trajectory, calculate_hd, smooth_ds
plt.ioff()

        
def run_movement(derivatives_base: Path, trials_to_include: list, conf_threshold: float = 0.6, show_plots: bool = False,  frame_rate: int = 25):
    """
    Processes the raw xy data as obtained from running inference.
    Creates:
    - XY_HD_t{tr}.csv files with xy position and hd
    - Plots of the position of keypoints before and after smoothing
    - Plots with animal trajectory (in animal trajectory folder)

    Inputs:
        derivatives_base (Path): path to derivatives folder
        trials_to_include (list): trials to look at 
        conf_threshold (float: 0.6): threshold used to remove values with confidence lower than this
        show_plots (bool: False): whether to show plots or just to save them
        frame_rate (int: 25): frame rate of camera. Defaults to 25
    
    Steps:
        1. Removes intervals that we want to remove
        2. Filters by confidence
        3. Interpolates (creates plot)
        4. Smoothes (creates plot)
        5. Calculates HD (orthogonal to ears and other methods, though other methods are not used)
        6. Plots trajectory
    
    Notes:
        If camera will be changed, x and y lims for the animal trajecotry plots have to be changed
        Assumes that the inference data has the format "T{tr}_*.h5'
    """

    output_folder, folder_directory, positional_data_folder, movement_data_folder, trajectory_data_folder = create_folder(derivatives_base)
    
    pixels_per_cm = get_pixels_per_cm(derivatives_base)
    
    for tr in trials_to_include:
        file_path = find_filepath(folder_directory, tr)

        # --------- Plotting confidence and position --------
        ds = load_poses.from_sleap_file(file_path, fps=frame_rate)
        
        #ds = remove_unneeded_intervals(ds, tr, derivatives_base)
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
    
        speed = calculate_speed(x, y, pixels_per_cm, frame_rate)
        #velocity = compute_velocity_movement(position, pixels_per_cm, frame_rate)
        speed_bool = speed >= 2
        speed_bool = speed_bool.astype(float)      # True→1.0, False→0.0
        speed_bool[np.isnan(speed)] = np.nan
        # HD that we'll be using: orthogonal to xy
        hd  = hd_orth_ears.values if hasattr(hd_orth_ears, "values") else hd_orth_ears

        # Save x, y, and hd to df
        save_to_df(x, y, hd, positional_data_folder,  f"XY_HD_t{tr}.csv", speed = speed_bool)


        # ------------- Plotting trajectory------------- 
        plot_trajectory(position, trajectory_data_folder, tr)
        
        # Saving center coordinates (used in consink code)
        x, y = get_xy(position, 'center')
        save_to_df(x, y, hd, positional_data_folder,  f"XY_HD_center_t{tr}.csv", speed = speed)
        
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
        csv_path = positional_data_folder/ f"XY_HD_skeleton_t{tr}.csv"
        df.to_csv(csv_path, index=False)
        print(f"→ Saved positional + HD data to {csv_path}")
        
        # ----------- Plot speed ------------
        plot_speed(speed, movement_data_folder, tr, frame_rate, show_plots)


if __name__ == "__main__":
    derivatives_base =  r"C:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    
    run_movement(derivatives_base, [2], show_plots = True)