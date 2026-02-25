from tracking_pipeline.run_movement import run_movement
from tracking_pipeline.overlay_video_HD import overlay_video_HD
from tracking_pipeline.combine_pos_csvs import combine_pos_csvs
from tracking_pipeline.add_platforms_to_csv import add_platforms_to_csv
from tracking_pipeline.plot_heatmap import plot_heatmap_xy
from maze_and_platforms.overlay_maze_image import overlay_maze_image
from maze_and_platforms.overlay_maze_image_consinks import overlay_maze_image_consinks
from maze_and_platforms.overlay_maze_outline import plot_maze_outline
from maze_and_platforms.get_limits import get_limits
from HCT_analysis.utilities.create_restricted_df import create_restricted_df
from HCT_analysis.utilities.restrict_posdata_specialbehav import restrict_posdata_specialbehav
from HCT_analysis.plotting.make_maze_plots import make_maze_behaviour_plots
from spatial_features.plot_ratemaps_and_hd import plot_ratemaps_and_hd
from spatial_features.plot_ratemaps_and_hd_pergoal import plot_ratemaps_and_hd_pergoal
from spatial_features.combine_autowv_ratemaps import combine_autowv_ratemaps
from spatial_features.plot_ratemaps_and_hd_speedfilt import plot_ratemaps_and_hd_speedfilt
from unit_features.plot_spikecount_over_trials import plot_spikecount_over_trials
from unit_features.test_restrict_spiketrain import test_restrict_spiketrain
from unit_features.get_spiketimes_allunits import get_spiketimes_alltrials
from unit_features.export_unit_spiketimes import export_unit_spiketimes
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use('TkAgg')
r"""
Run this script in your movement environment (movement-env2 for Sophia and )

Note: create this environment in conda
conda create -n movement-env -c conda-forge movement napari pyqt"""


tasks_to_run = [5]
# 1 : Initial HCT plotting and creating restricted df
# 2 : Getting platform locations through overlay and the plotting maze outline and limits
# 3 : Running movement and making overlay video
# 4 : Making ratemaps and HD plots, spikecount over trials, and combining these
# 5 : Speed filtering

frame_rate = 25
trials_to_include= np.arange(1,14)
print(trials_to_include)
derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-02_date-12022026\second_run_1602"
derivatives_base = Path(derivatives_base)
task = 'hct' # hct or spatiotemp
goals_to_include = [1] # This is for the HCT. 
# 0: animal going to g2 during g1
# 1: animal going to g1
# 2: animal going to g2p


good_overlay = "y"
unit_type = "all" # all, pyramidal, or good
run_which_units_hct = "all" # all, pyramidal, good, or test (first 5 units)
show_plots = False
save_plots = True
include_open_field = True
clear_plot_folder = False
short_overlay = True


####### INITIAL HCT PLOTTING AND CSV CREATION ##############
if task == "hct" and 1 in tasks_to_run:
    # Takes the alltrials csv and creates a new csv only with the rows of the trial date
    create_restricted_df(derivatives_base, goals_to_include, trials_to_include)

    # Plots occupancy, start platforms, and proportion correct
    make_maze_behaviour_plots(derivatives_base, goals_to_include,  show_plots = show_plots)


############ GETTING PLATFORM LOCATIONS AND PLOTTING PARAMS #############
# Add platform coordinates to the center positional csv
if 2 in tasks_to_run:
    good_overlay, img = overlay_maze_image(derivatives_base,  method = "video")

    # Obtains limits
    get_limits(derivatives_base)

    # Plots maze outline
    plot_maze_outline(derivatives_base, img)

    if task == "hct":
        overlay_maze_image_consinks(derivatives_base, method = "video")
########### TRACKING AND COMBINING POS DATA ############
# run_movement gives us the xy coordinates and hd
if 3 in tasks_to_run:
    #run_movement(derivatives_base,trials_to_include = trials_to_include, conf_threshold = 0.4, frame_rate = frame_rate, show_plots= show_plots)

    # overlays the hd on the video
    #overlay_video_HD(derivatives_base, [4, 5, 6], short = short_overlay)

    # combines all the positional csvs. Output: XY_HD_alltrials.csv, XY_HD_alltrials_center.csv (gives the center coordinates)
    combine_pos_csvs(derivatives_base, trials_to_include, frame_rate =frame_rate)


    platforms_path = derivatives_base/"analysis"/"spatial_behav_data"/"XY_and_HD"/'XY_HD_w_platforms.csv'
    if not platforms_path.exists():
        add_platforms_to_csv(derivatives_base, file= 'center', frame_rate = frame_rate)

    if task == "hct":
        restrict_posdata_specialbehav(derivatives_base, goals_to_include, show_plots = show_plots)
        platforms_path_intervals = derivatives_base/"analysis"/"spatial_behav_data"/"XY_and_HD"/'XY_HD_allintervals_w_platforms.csv'
        if not platforms_path_intervals.exists():
            add_platforms_to_csv(derivatives_base, file= 'intervals')



############ GETTING SPATIAL AND UNIT FEATURES #############
# Rate map + hd for each unit
if 4 in tasks_to_run:
    plot_ratemaps_and_hd(derivatives_base, unit_type = unit_type, save_plots = save_plots, show_plots = show_plots, clear_plot_folder = clear_plot_folder, frame_rate= frame_rate)

    # Spikecount over time for each unit
    plot_spikecount_over_trials(derivatives_base, unit_type = unit_type, trials_to_include = trials_to_include, task = task, last_trial_openfield = include_open_field)

    # Combines autocorrelogram + wv, ratemap + hd, and spikecount over tiem map
    combine_autowv_ratemaps(derivatives_base, unit_type = unit_type, rmap_per_goal= False)


    if task == "hct":

        plot_heatmap_xy(derivatives_base, trials_to_include, goals_to_include, frame_rate)

        plot_ratemaps_and_hd_pergoal(derivatives_base, unit_type = unit_type, goals_to_include= goals_to_include, include_open_field = include_open_field, save_plots= save_plots, show_plots= show_plots,clear_plot_folder= clear_plot_folder, frame_rate = frame_rate)
        
        combine_autowv_ratemaps(derivatives_base, unit_type = unit_type, rmap_per_goal= True)

        test_restrict_spiketrain(derivatives_base, unit_type = unit_type, goals_to_include = goals_to_include, show_plots = show_plots)


########## SPEED FILTERING ################
if 5 in tasks_to_run:
    # Get spiketimes for each unit, filtered by speed
    speed_threshold = 2 #default: 2cm/s
    export_unit_spiketimes(derivatives_base, goals_to_include=goals_to_include, add_speed_filt = True, speed_threshold = speed_threshold, frame_rate = frame_rate)
    
    # exports spiketimes of all units for each trial, filtered by speed
    #get_spiketimes_alltrials(derivatives_base, speed_filt = True, frame_rate = frame_rate)
    show_plots = True
    include_open_field = False
    # Plot ratemaps for the speed filtered values
    plot_ratemaps_and_hd_speedfilt(derivatives_base, unit_type = "pyramidal", goals_to_include=goals_to_include, include_open_field=include_open_field,
                                   save_plots = save_plots, show_plots = show_plots, clear_plot_folder=clear_plot_folder, frame_rate = frame_rate)
