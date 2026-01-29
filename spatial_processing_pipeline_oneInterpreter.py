import numpy as np
from tracking_pipeline.run_movement import run_movement
from tracking_pipeline.overlay_video_HD import overlay_video_HD
from tracking_pipeline.combine_pos_csvs import combine_pos_csvs
from maze_and_platforms.overlay_maze_image_fromVideo import overlay_maze_image
from maze_and_platforms.overlay_maze_image_consinks import overlay_maze_image_consinks
from tracking_pipeline.add_platforms_to_csv import add_platforms_to_csv
from maze_and_platforms.overlay_hexagonal_grid import plot_maze_outline
from spatial_features.make_spatiotemp_plots import make_spatiotemp_plots
from spatiotemporal_analysis.make_epoch_times_csv import make_epoch_times_csv
from HCT_analysis.get_limits import get_limits
from HCT_analysis.utilities.concat_trials import concat_trials
from HCT_analysis.utilities.restrict_posdata_specialbehav import restrict_posdata_specialbehav
from HCT_analysis.utilities.create_intervals_specialbehav import create_intervals_specialbehav, check_restricted_df_exists, make_restricted_df
from HCT_analysis.utilities.trials_utils import append_alltrials, get_goal_numbers
from HCT_analysis.plotting.make_maze_plots import plot_occupancy, plot_propcorrect, plot_startplatforms
from HCT_analysis.plotting.plot_intervals import plot_intervals
from HCT_analysis.find_consinks_main import main as find_consinks_main
from spatial_features.plot_ratemaps_and_hd import plot_ratemaps_and_hd
from unit_features.plot_firing_each_epoch import plot_firing_each_epoch
from HCT_analysis.get_directional_occupancy_by_pos import main as directional_occupancy_main

#from unit_features.plot_spikecount_over_trials import plot_spikecount_over_trials
#from spatial_features.get_spatial_features import get_spatial_features
from spatial_features.roseplot import make_roseplots
from spatial_features.HD_across_epoch import sig_across_epochs
from spatial_features.combine_autowv_ratemaps import combine_autowv_ratemaps
import os
from unit_features.plot_spikecount_over_trials import plot_spikecount_over_trials
from HCT_analysis.calculate_vectorfields_newmethod import main as calculate_vectorfields
import matplotlib
matplotlib.use('TkAgg')
r"""
Run this script in your movement environment (movement-env2 for Sophia and )
Inputs:
derivatives_base: derivatives folder path (e.g. r"D:\Spatiotemporal_task\derivatives\sub-002_id-1U\ses-01_date-02072025\all_trial")
trials_to_include: array with trial number
centroid_model_folder: path to centroid model
centered_model_folder: path to centered model


Note: create this environment in conda
conda create -n movement-env -c conda-forge movement napari pyqt"""

trials_to_include= np.arange(1,27)
print(trials_to_include)
derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
task = 'hct' # hct or spatiotemp
goals_to_include = [1] # This is for the HCT. 
# 0: animal going to g2 during g1
# 1: animal going to g1
# 2: animal going to g2
good_overlay = "y"
run_which_units = "all" # all, pyramidal, or good
run_which_units_hct = "test" # all, pyramidal, good, or test (first 5 units)
show_plots = False
if task == "hct":
    # Takes the alltrials csv and creates a new csv only with the rows of the trial date
    append_alltrials(derivatives_base)
    
    """
    restricted_df_exists =  check_restricted_df_exists(derivatives_base)
    if restricted_df_exists:
        print("Restricted_df exists")
    if not restricted_df_exists and 0 not in goals_to_include:
        make_restricted_df(derivatives_base, goals_to_include, trials_to_include)
        create_intervals_specialbehav(derivatives_base)
        #restrict_posdata_specialbehav(derivatives_base, goals_to_include)
    elif restricted_df_exists:
        create_intervals_specialbehav(derivatives_base)
        #restrict_posdata_specialbehav(derivatives_base, goals_to_include)
    else:
        print("0 in goals to include and restricted_df.csv doesn't exist. Please create manually and rerun")

    concat_trials(derivatives_base)
    goal_platforms = get_goal_numbers(derivatives_base)
    # Shows a maze plot with the occupancy of each platform for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_occupancy(derivatives_base, goal_platforms, goals_to_include=goals_to_include, show_plots= show_plots)

    # Shows a maze plot with all the start platforms
    plot_startplatforms(derivatives_base, goal_platforms, show_plots= show_plots, goals_to_include=goals_to_include)

    # Shows a maze plot with the proportion of correct choices for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_propcorrect(derivatives_base,goal_platforms, goals_to_include=goals_to_include, show_plots= show_plots)

# Add platform coordinates to the center positional csv
good_overlay, img = overlay_maze_image(derivatives_base,  method = "video")

get_limits(derivatives_base)

plot_maze_outline(derivatives_base, img)

# run_movement gives us the xy coordinates and hd
run_movement(derivatives_base,trials_to_include, conf_threshold = 0.4, frame_rate = 25, show_plots= show_plots)
"""
# overlays the hd on the video
overlay_video_HD(derivatives_base, [18], short = False)


# combines all the positional csvs. Output: XY_HD_alltrials.csv, XY_HD_alltrials_center.csv (gives the center coordinates)
combine_pos_csvs(derivatives_base, trials_to_include, frame_rate =25)
restricted_df_exists = False
if task == "hct" and restricted_df_exists and not os.path.exists(os.path.join(derivatives_base, "analysis", "spatial_behav_data", "XY_and_HD", "XY_HD_allintervals_w_platforms.csv")):
    add_platforms_to_csv(derivatives_base, file= 'intervals')

if not os.path.exists(os.path.join(derivatives_base, "analysis", "spatial_behav_data", "XY_and_HD", 'XY_HD_w_platforms.csv')):
    add_platforms_to_csv(derivatives_base, file= 'center')

if False:
    # Rate map + hd for each unit
    plot_ratemaps_and_hd(derivatives_base, unit_type = 'all')

    # NOTE: add ratemaps per goal

    # Spikecount over time for each unit
    plot_spikecount_over_trials(derivatives_base, 'all', trials_to_include, task = task)


    # Combines autocorrelogram + wv, ratemap + hd, and spikecount over tiem map
    combine_autowv_ratemaps(derivatives_base, unit_type = 'all')

if good_overlay == 'y' and task == 'hct':
    
    
    ## PLOTTING
    # Concatenates all trial files in the form {ratID}_{date}_g{trial}.csv into one df called concatenated_trials.csv
    concat_trials(derivatives_base)
    
    print("You want to go past this point only once you've classified the cells\n Create pyramidal.. 2D .csv and then continue (c)")

    directional_occupancy_main(derivatives_base, goals_to_include = goals_to_include)
    # Find consinks
    find_consinks_main(derivatives_base, rel_dir_occ='intervals',unit_type= "test",  goals_to_include = goals_to_include, show_plots = False, code_to_run=[ 2,3,4])

    # Find vectorfields
    #calculate_vectorfields(derivatives_base, goals_to_include = goals_to_include)

    # Find popsink
    pass
elif task == 'spatiotemp':
    make_epoch_times_csv(derivatives_base, trials_to_include)
    plot_firing_each_epoch(derivatives_base, trials_to_include, unit_type = "all")
    degrees_df_path, deg = make_spatiotemp_plots(derivatives_base,trials_to_include, unit_type = "all", make_plots = False)
    make_roseplots(derivatives_base, trials_to_include, deg, path_to_df = degrees_df_path)
    sig_across_epochs(derivatives_base, trials_to_include)
else:
    print("Overlay parameters were not accepted. Platforms will not be assigned to csv file.")
    print("Adjust parameters in overlay_maze_image_fromVideo function to continue analysis.")
    print("You can use maze_and_platforms\find_hexagon.ipynb for easier finding of the parameters")