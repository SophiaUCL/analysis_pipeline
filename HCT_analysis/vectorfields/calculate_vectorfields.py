import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import pandas as pd
import sys
sys.path.append('C:/Users/Jake/Documents/python_code/rob_maze_analysis_code')
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.trials_utils import get_goal_coordinates
from utilities.mrl_func import resultant_vector_length
from astropy.stats import circmean
cm_per_pixel = 1


def calculate_vector_fields(spike_rates_by_position_and_direction):

    bin_centres = spike_rates_by_position_and_direction['direction_bins'][:-1] + np.diff(spike_rates_by_position_and_direction['direction_bins'])/2

    vector_fields = {'units': {}, 'x_bins': spike_rates_by_position_and_direction['x_bins'], 
                     'y_bins': spike_rates_by_position_and_direction['y_bins'], 
                     'direction_bins': spike_rates_by_position_and_direction['direction_bins']}
    mean_resultant_lengths = {'units': {}, 'x_bins': spike_rates_by_position_and_direction['x_bins'], 
                     'y_bins': spike_rates_by_position_and_direction['y_bins'], 
                     'direction_bins': spike_rates_by_position_and_direction['direction_bins']} 
    

    # poss_unit_keys = ['units', 'popn']
    # data_keys = list(spike_rates_by_position_and_direction.keys())
    # # find the key common to both lists
    # unit_key = [k for k in poss_unit_keys if k in data_keys][0]

    spike_rates_by_position_and_direction = spike_rates_by_position_and_direction['units']
    units = list(spike_rates_by_position_and_direction.keys())

    for u in units:
        rates_by_pos_dir = spike_rates_by_position_and_direction[u]
        array_shape = rates_by_pos_dir.shape

        # initialize vector field as array of nans
        vector_field = np.full(array_shape[0:2], np.nan)
        mrl_field = np.full(array_shape[0:2], np.nan)

        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                rates = rates_by_pos_dir[i, j, :]
                
                # if any nan values in rates, OR if all values are zero skip
                if np.isnan(rates).any() or np.all(rates == 0):
                    continue

                mean_dir = np.round(circmean(bin_centres, weights = rates), 3)
                mrl = np.round(resultant_vector_length(bin_centres, w = rates), 3)

                if mean_dir > np.pi:
                    mean_dir = mean_dir - 2*np.pi

                vector_field[i, j] = mean_dir
                mrl_field[i, j] = mrl

        vector_fields['units'][u] = vector_field
        mean_resultant_lengths['units'][u] = mrl_field

    return vector_fields, mean_resultant_lengths






def plot_vector_fields_all(unit_ids, vector_fields, goal_coordinates, x_centres, y_centres, plot_dir, consink_df):
    """Plot vector fields for all units. 3x1 plot. First plot: all data, second plot: goal 1 data, third plot: goal 2 data. 
    Saves plots in plot_dir for each unit. 

    Args:
        unit_ids (list): List of unit IDs to plot.
        vector_fields (dict): Dictionary of vector fields for each unit.
        goal_coordinates (list): List of goal coordinates.
        x_centres (np.ndarray): X coordinates of the bins.
        y_centres (np.ndarray): Y coordinates of the bins.
        plot_dir (str): Directory to save plots.
        consink_df (pd.DataFrame): DataFrame containing consistency sink information.
    """

    # Create plot directory if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for u in unit_ids:
        fig, ax = plt.subplots(1, 3, figsize=(20, 10))
        fig.suptitle(f"Vector fields unit {u}", fontsize=24)

        goals = [0,1,2]

        for i, g in enumerate(goals):
            if g == 0:
                ax[i].set_title('all trials', fontsize=20)
            else:
                ax[i].set_title(f'goal {g}', fontsize=20)
            ax[i].set_xlabel('x position (cm)', fontsize=16)
            ax[i].set_ylabel('y position (cm)', fontsize=16)

            # plot the goal positions
            if g > 0:
                circle = plt.Circle((goal_coordinates[g-1][0], 
                        goal_coordinates[g-1][1]), 80, color='g', 
                        fill=False, linewidth=5)
                ax[i].add_artist(circle)    
            
            try:
                vector_field = vector_fields[g]['units'][int(u)]
            except:
                print(f"No vector field for unit {u} and goal {g}")
                continue

            # plot vector field
            ax[i].quiver(x_centres, y_centres, np.cos(vector_field), np.sin(vector_field), color='k', scale=10)
        
            # flip y axis
            ax[i].invert_yaxis()

            # increase the range of the axes by 10% to make room for the arrows
            x_lim = ax[i].get_xlim()
            y_lim = ax[i].get_ylim()
            x_range = x_lim[1] - x_lim[0]
            y_range = y_lim[1] - y_lim[0]
            ax[i].set_xlim(x_lim[0] - 0.1*x_range, x_lim[1] + 0.1*x_range)
            ax[i].set_ylim(y_lim[0] - 0.1*y_range, y_lim[1] + 0.1*y_range)

            if g > 0:
                consink_row = consink_df.loc[u]

                mrl = consink_row[f'mrl_g{g}']
                consink_pos = consink_row[f'position_g{g}']
                consink_angle = consink_row[f'mean_angle_g{g}']
                ci_95 = consink_row[f'ci_95_g{g}']
                if consink_angle > np.pi:
                    consink_angle = consink_angle - 2*np.pi
                
                # plot a filled circle at the consink position
                if mrl > ci_95:
                    consink_color = 'r'
                else: # color is gray
                    consink_color = 'gray'

                circle = plt.Circle((consink_pos[0], 
                    consink_pos[1]), 50, color=consink_color, 
                    fill=True)
                ax[i].add_artist(circle)      
            
                # add text with mrl, ci_95, ci_999
                ax[i].text(400, 1800, f'mrl: {mrl:.2f}\nci_95: {ci_95:.2f}\nangle: {np.rad2deg(consink_angle):.2f}', fontsize=16)
                #ax[i].text(0, 2100, f'mrl: {mrl:.2f}\nci_999: {ci_999:.2f}\nangle: {consink_angle:.2f}', fontsize=16)
                
            # set font size of axes
            ax[i].tick_params(axis='both', which='major', labelsize=14)
            """
            # get the axes values
            x_ticks = ax[i].get_xticks()
            y_ticks = ax[i].get_yticks()

            # convert the axes values to cm
            x_ticks_cm = x_ticks * cm_per_pixel
            y_ticks_cm = y_ticks * cm_per_pixel

            # set the axes values to cm
            ax[i].set_xticklabels(x_ticks_cm)
            ax[i].set_yticklabels(y_ticks_cm)

            """

            # set the axes to have identical scales
            ax[i].set_aspect('equal')        

        fig.savefig(os.path.join(plot_dir, f'vector_fields_unit_{u}.png'))
        plt.close(fig)
        
        
def main(derivatives_base, rawsession_folder):
    """
    For each unit, creates a 3x1 subplot of vector fields for all trials, goal 1 trials, and goal 2 trials.
    run get_directional_occupancy_by_pos.py first to generate the necessary data.

    Args:
        derivatives_base (str): The base directory for the derivatives.
        
    Note! Goal coordinates is not correctly defined yet
    """
    
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])


    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'spatial_features', 'consink_data')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    goal_coordinates = get_goal_coordinates(derivatives_base, rawsession_folder)
    consink_df = load_pickle('consinks_df', output_folder)
    
    
    ############# CALCULATING VECTOR FIELDS ##############3
    vector_fields = {}
    mean_resultant_lengths = {}
    
    # First for all trials
    spike_rates_by_position_and_direction = load_pickle('spike_rates_by_position_and_direction', output_folder)
    vector_fields[0], mean_resultant_lengths[0] = calculate_vector_fields(spike_rates_by_position_and_direction)

    # Now per goal
    for g in [1,2]:
        print("Calculating vector fields for goal ", g)
        spike_rates_by_position_and_direction_by_goal = load_pickle( f'spike_rates_by_position_and_direction_g{g}', output_folder)
        vector_fields[g], mean_resultant_lengths[g] = calculate_vector_fields(spike_rates_by_position_and_direction_by_goal)
    print("Saving")
    # Save
    save_pickle(vector_fields, 'vector_fields', output_folder)
    save_pickle(mean_resultant_lengths, 'mean_resultant_lengths', output_folder)
    
    # Plotting
    plot_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'spatial_features', 'vector_fields')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    print(f"Saving to {plot_dir}")
    x_bins = vector_fields[0]['x_bins']
    x_centres = x_bins[:-1] + np.diff(x_bins)/2
    y_bins = vector_fields[0]['y_bins']
    y_centres = y_bins[:-1] + np.diff(y_bins)/2
    plot_vector_fields_all(unit_ids, vector_fields, goal_coordinates, x_centres, y_centres, plot_dir, consink_df)

    consink_csv_path = os.path.join(output_folder, 'consinks_df.csv')
    consink_df.to_csv(consink_csv_path, index=True)
    print(f"Saved consink_df as CSV to: {consink_csv_path}")
    



if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    main(derivatives_base, rawsession_folder)










    