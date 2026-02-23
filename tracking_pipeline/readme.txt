Folder: tracking_pipeline

Contains files used to track the animal location. Files are called from spatial_processing_pipeline

Specific files:
add_platforms_to_csv: Adds platforms to XY_HD_alltrials_center.csv and saves it as XY_HD_w_platforms.csv (if file == center)
    if file == intervals, then we take the file 'XY_HD_allintervals.csv' and add platforms to it.
Exports: 'XY_HD_w_platforms.csv' if file == 'center' else 'XY_HD_allintervals_w_platforms.csv'

combine_pos_csvs: Combines all data from XY_HD_t{tr}.csv (for tr in trials_to_include) into one csv called HD_XY_alltrials.csv 
and saves it in the same folder as the XY_HD_t{tr}.csvs
Outputs:
derivatives_base\analysis\spatial_behav_data\XY_and_HD\XY_HD_alltrials.csv: all the trials concatenated together with padding inbetween
        trials to match trial length. xy position corresponds to center of ears.
        derivatives_base\analysis\spatial_behav_data\XY_and_HD\XY_HD_alltrials_center.csv: same as before but xy is the center of the animal


overlay_video_HD: Overlays videos with the position of the animal and the head directio
Outputs:
derivatives_base\analysis\processed_video\T{tr}_with_HD.avi: video file with overlayed position and HD

plot_heatmap_xy: Makes a heatmap of the xy position of the animal for each trial. 
    This is used to check whether all xy positions are correct
Outputs: plots per trial and for all trials in spatial_behav_data/position_heatmaps

run_movement: Processes the raw xy data as obtained from running inference.
Creates:
    - XY_HD_t{tr}.csv files with xy position and hd
    - Plots of the position of keypoints before and after smoothing
    - Plots with animal trajectory (in animal trajectory folder)

run_movement_utils: Utility functions used for run_movement

