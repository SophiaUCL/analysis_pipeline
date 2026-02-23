import numpy as np
import os
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
from pathlib import Path
import shutil
print("RUNNING FILE:", os.path.abspath(__file__))
from spatial_features.utils.spatial_features_utils import add_relative_hd, get_goal_coordinates, get_goal_numbers, get_ratemaps_restrictedx, load_unit_ids, get_outline, get_limits, get_posdata, get_occupancy_time, get_ratemaps, get_spike_train_frames, get_directional_firingrate
from spatial_features.utils.spatial_features_plots import plot_rmap, plot_occupancy, plot_directional_firingrate
from spatial_features.utils.restrict_spiketrain_specialbehav import get_spike_train, restrict_spiketrain_specialbehav


UnitTypes = Literal['pyramidal', 'good', 'all']

def plot_ratemaps_and_hd_pergoal(derivatives_base: Path, unit_type: UnitTypes, goals_to_include: np.ndarray | list = [0,1,2], include_open_field: bool = False, save_plots: bool=True, show_plots= False,clear_plot_folder: bool = False, frame_rate: int = 25, sample_rate: int = 30000):
    """ 
    Makes a plot for each unit with its ratemap per goal
    
    Inputs
    -------
    derivatives_base (Path): Path to derivatives folder
    unit_type (pyramidal, good, or all): units for which the plots will be made
    goals_to_include (ndarray | list: [0,1,2]): array with goal numbers that were run for these recordings
    save_plots (bool: True): whether plots are saved to the folder
    show_plots (bool: False): whether plots are displayed
    clear_plot_folder (bool: False): whether to clear all the plots in the plot folder (set to True after merging)
    frame_rate (int: 25): frame rate of camera
    sample_rate (int: 30000): sample rate of recording
    
    Saves
    ---------
    derivatives_base/'analysis'/ 'cell_characteristics'/'spatial_features'/ 'ratemaps_and_hd_allgoals'
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Load data files
    kilosort_output_path = derivatives_base/ 'ephys'/ "concat_run"/"sorting"/ "sorter_output" 
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    
    # Output folder
    output_folder = derivatives_base/'analysis'/ 'cell_characteristics'/'spatial_features'/ 'ratemaps_and_hd_allgoals'
    if clear_plot_folder:
        print("Clearing output folder")
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents = True, exist_ok = True)

    # Here unit ids get filtered by unit_type
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    
    # Limits and outline
    xmin, xmax, ymin, ymax = get_limits(derivatives_base)
    outline_x, outline_y = get_outline(derivatives_base)
    
    goals = np.append(goals_to_include, 3) # g == 3 corresponds to the whoel trial
    if include_open_field:
        goals = np.append(goals, 4) # g == 4 corresponds to open field
    
    x_allg = []
    y_allg = []
    hd_allg = []
    occupancy_time_allg = []
    rel_occupancy_time_allg = []
    num_bins = 24
    
    goal_coordinates = get_goal_coordinates(derivatives_base, rawsession_folder)
    
    method = "ears"
    add_relative_hd(derivatives_base, goal_coordinates, method = method,  goals = [el for el in goals if el != 0])
    
    threshold = [0.6,1,0.4] # THRESHOLD FOR RMAP OCCUPANCY
    
    
    for g in goals:
        # Get directory for the positional data
        x, y, hd,_ = get_posdata(derivatives_base, method = method, g = g)

        # Obtaining hd for this trial how much the animal sampled in each bin
        occupancy_time = get_occupancy_time(hd, frame_rate, num_bins = num_bins)
        x_allg.append(x)
        y_allg.append(y)
        hd_allg.append(hd)
        occupancy_time_allg.append(occupancy_time)
        
        if g == 3:
            x_fulltrial = x
            y_fulltrial = y
            hd_fulltrial = hd
    
    goals_without_g03 = [g for g in goals if g !=0 and g !=3 and g != 4]
    
    for g in goals_without_g03:
        data_path = derivatives_base/'analysis'/'spatial_behav_data'/'XY_and_HD'/f'XY_HD_goal{g}_trials.csv'
        relative_hd = pd.read_csv(data_path).iloc[:, 3].to_numpy()
        rel_occupancy_time = get_occupancy_time(relative_hd, frame_rate, num_bins = num_bins)
        rel_occupancy_time_allg.append(rel_occupancy_time)
    
    for g in goals_without_g03:
        if method == "ears":
            fulltrial_path = derivatives_base/'analysis'/'spatial_behav_data'/'XY_and_HD'/'XY_HD_alltrials.csv'
        elif method == "center":
            fulltrial_path = derivatives_base/'analysis'/'spatial_behav_data'/'XY_and_HD'/'XY_HD_alltrials_center.csv'
        pos_data= pd.read_csv(fulltrial_path)
        if g == 1:
            relhd_fulltrial_g1 = pos_data[f'relative_hd_g{g}'].to_numpy()
        elif g == 2:
            relhd_fulltrial_g2 = pos_data[f'relative_hd_g{g}'].to_numpy()
        
            
    print("Plotting ratemaps and hd")
    print(f"Saving results to {output_folder}")
    for unit_id in tqdm(unit_ids):
        # Make plot
        fig, axs = plt.subplots(2, len(goals), figsize = [len(goals)*5, 10])
        fig.suptitle(f"Unit {unit_id}", fontsize = 18)
        for column, g in enumerate(goals):
            
            if g == 2 or g ==1 :
                name = f"Goal {g}"
            elif g == 3:
                name = "Full recording"
            elif g == 4:
                name = "Open Field"
            x_g = x_allg[column]
            y_g = y_allg[column]
            occupancy_time_g = occupancy_time_allg[column]
            #rel_occupancy_time_g = rel_occupancy_time_allg[column -1] if g !=0 and g !=3 and g != 4 else None
            threshold_occ = threshold[column ]
            if g == 1:
                relhd_fulltrial = relhd_fulltrial_g1
            elif g == 2:
                relhd_fulltrial = relhd_fulltrial_g2
            
            if g < 4:
                # Load spike data for this goal in frames
                spike_train = get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, x_g)
            else:
                spike_train = get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, x_fulltrial)


            
            # ===== Plot ratemap ====

            rmap, x_edges, y_edges=  get_ratemaps_restrictedx(spike_train, x_fulltrial, y_fulltrial, x_g, y_g, occupancy_threshold = threshold_occ)
            plot_rmap(rmap, xmin, xmax, ymin, ymax, x_edges, y_edges, outline_x, outline_y, ax = axs[0, column], fig = fig, title = f"{name}, n = {len(spike_train)}")

        
            # === Plot HD ===
            axs[1, column].remove()
            axs[1, column] = fig.add_subplot(2, len(goals), len(goals) + column + 1, projection="polar")



            direction_firing_rate, bin_centers = get_directional_firingrate(hd_fulltrial, spike_train, num_bins, occupancy_time_g)
            
            #fig.delaxes(axs[1,column])
            #axs[1, column] = fig.add_subplot(2, len(goals), len(goals) + column + 1, polar=True)
            plot_directional_firingrate(bin_centers, direction_firing_rate, ax = axs[1,column])

            
            """
             axs[2, column].remove()
            axs[2, column] = fig.add_subplot(3, len(goals), 2*len(goals) + column + 1, projection="polar")

            if g == 1 or g == 2:
                # === Plot relative HD ===
                direction_firing_rate_rel, bin_centers = get_directional_firingrate(relhd_fulltrial, spike_train, num_bins,  rel_occupancy_time_g)
                
                #fig.delaxes(axs[2,column])
                #axs[2, column] = fig.add_subplot(3, len(goals), len(goals) + column + 1, polar=True)

                plot_directional_firingrate(bin_centers, direction_firing_rate_rel, ax = axs[2,column])
            """

        output_path = output_folder/ f"unit_{unit_id}_rm_hd.png"
        if save_plots:
            plt.savefig(output_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        
    
    print(f"Saved plots to {output_folder}")
        
if __name__ == "__main__":    
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
    plot_ratemaps_and_hd_pergoal(derivatives_base,unit_type = "pyramidal", include_g0 = False, saveplots=True, show_plots=True)



