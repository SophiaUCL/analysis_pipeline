import numpy as np
import os
import sys
from utilities.concat_trials import concat_trials
from utilities.create_intervals import create_intervals_df
from utilities.trials_utils import append_alltrials, get_goal_numbers
from plotting.make_maze_plots import plot_occupancy, plot_propcorrect, plot_startplatforms
from plotting.plot_intervals import plot_intervals
from get_limits import get_limits
from find_consinks import main as find_consinks
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from display_maze.overlay_maze_image import overlay_maze_image
from get_directional_occupancy_by_pos import main as directional_occupancy_main
from calculate_vectorfields import main as calculate_vectorfields
#from find_popsink import calculate_popsink
import matplotlib
matplotlib.use('Qt5Agg') 

# Running HCT analysis
def run_HCT(rawsession_folder, derivatives_base):
    """
    NOTE 27/01/26: OLD, currently not in use, though does give  a nice overview
    Combination of all functions needed for HCT analysis
    
    Maze plots can be found in {derivatives_base}/analysis/maze_behaviour
    """
    print("entered")
    # overlay maze image
    #overlay_maze_image(derivatives_base) # Currently we're doing this during movement, so don't have to do that now
    # Concatenates all trial files in the form {ratID}_{date}_g{trial}.csv into one df called concatenated_trials.csv

    concat_trials(rawsession_folder)
    
    # Takes the alltrials csv and creates a new csv only with the rows of the trial date
    append_alltrials(rawsession_folder)
    
    # Find goal numbers. NOTE: add here to make json with goal_platofmrs and goal coordinates
    goal_platforms = get_goal_numbers(rawsession_folder)
    
    # NOTE: here we should create the intervals, but we should use the new code.add()
    
    ############ Plotting ###########
    # Shows a maze plot with the occupancy of each platform for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_occupancy(derivatives_base, rawsession_folder, goal_platforms)

    # Shows a maze plot with all the start platforms
    plot_startplatforms(derivatives_base, rawsession_folder, goal_platforms)

    # Shows a maze plot with the proportion of correct choices for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_propcorrect(derivatives_base, rawsession_folder, goal_platforms)
    ################################
    ### CONSINK ANALYSIS ###
    
    # Plots the intervals to ensure that they are made correctly
    # NOTE: still to finish
    #plot_intervals(derivatives_base, rawsession_folder)
    
    # NOTE: Still to create
    #plot_spikes_over_intervals(derivatives_base, rawsession_folder)
    
    
    # Obtains coordinates for the square in which consinks will be placed
    get_limits(derivatives_base, rawsession_folder) 
    
    # ADd function that creates goal1 and goal2 pos data df
    
    # Gets directional occupancy per bin
    # Currently working! Hopefully no errors
    directional_occupancy_main(derivatives_base, rawsession_folder, code_to_run = [0,1])
     
    # NOTE: Still have to add coordinates here of goals
    find_consinks(derivatives_base, rawsession_folder)
    
    # Calculate snad plots vectorfields.
    calculate_vectorfields(derivatives_base)
    
    # NOTE: Quite a lot to fix here
    #calculate_popsink(derivatives_base)
    
    
if __name__ == "__main__":

    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    run_HCT(rawsession_folder, derivatives_base)
    