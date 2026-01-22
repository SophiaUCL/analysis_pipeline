import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
import warnings
print("RUNNING FILE:", os.path.abspath(__file__))
from utils.spatial_features_utils import add_relative_hd, get_goal_coordinates, get_goal_numbers, get_ratemaps_restrictedx, load_unit_ids, get_outline, get_limits, get_posdata, get_occupancy_time, get_ratemaps, get_spike_train_frames, get_directional_firingrate
from utils.spatial_features_plots import plot_rmap, plot_occupancy, plot_directional_firingrate
from utils.restrict_spiketrain_specialbehav import get_spike_train, restrict_spiketrain_specialbehav


    
def plot_ratemaps_and_hd_pergoal(derivatives_base, unit_type: Literal['pyramidal', 'good', 'all'], include_g0 = True, frame_rate = 25, sample_rate = 30000, saveplots=True, show_plots= False):
    """ 
    Makes a plot for each unit with its ratemap (left) and directional firing rate (right)

    Inputs: derivatives base
    
    """
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, 'ephys', "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    
    # Output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'ratemaps_and_hd_allgoals')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Here unit ids get filtered by unit_type
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    
    # Limits and outline
    xmin, xmax, ymin, ymax = get_limits(derivatives_base)
    outline_x, outline_y = get_outline(derivatives_base)
    
    goals = [0,1,2,3] if include_g0 else [1,2,3] # goal 3 is the full trial
    
    x_allg = []
    y_allg = []
    hd_allg = []
    occupancy_time_allg = []
    rel_occupancy_time_allg = []
    num_bins = 24
    
    goal_numbers = get_goal_numbers(derivatives_base)
    goal_coordinates = get_goal_coordinates(derivatives_base, rawsession_folder)
    
    method = "ears"
    add_relative_hd(derivatives_base, goal_coordinates, method = method,  goals = [1,2, 3])
    
    for g in goals:
        # Get directory for the positional data
        x, y, hd,_ = get_posdata(derivatives_base, method = method, g = g)

        # Obtaining hd for this trial how much the animal sampled in each bin
        occupancy_time = get_occupancy_time(hd, frame_rate, num_bins = num_bins)
        x_allg.append(x)
        y_allg.append(y)
        hd_allg.append(hd)
        occupancy_time_allg.append(occupancy_time)
    
    
     # Loop over units
    x_fulltrial = x_allg[-1]
    y_fulltrial = y_allg[-1]
    hd_fulltrial = hd_allg[-1]
    occupancy_time_fulltrial = occupancy_time_allg[-1]
    
    goals_without_g03 = [g for g in goals if g !=0 and g !=3]
    
    for g in goals_without_g03:
        data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{g}_trials.csv')
        relative_hd = pd.read_csv(data_path).iloc[:, 3].to_numpy()
        rel_occupancy_time = get_occupancy_time(relative_hd, frame_rate, num_bins = num_bins)
        rel_occupancy_time_allg.append(rel_occupancy_time)
    
    for g in goals_without_g03:
        if method == "ears":
                fulltrial_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials.csv')
        elif method == "center":
            fulltrial_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_alltrials_center.csv')
        pos_data= pd.read_csv(fulltrial_path)
        if g == 1:
            relhd_fulltrial_g1 = pos_data[f'relative_hd_g{g}'].to_numpy()
        elif g == 2:
            relhd_fulltrial_g2 = pos_data[f'relative_hd_g{g}'].to_numpy()
        
            
    print("Plotting ratemaps and hd")
    for unit_id in tqdm(unit_ids):
        # Make plot
        fig, axs = plt.subplots(3, len(goals), figsize = [len(goals)*5, 10])
        fig.suptitle(f"Unit {unit_id}", fontsize = 18)
        for g in goals:
            column = g if include_g0 else g -1
            
            x_g = x_allg[column]
            y_g = y_allg[column]
            hd_g = hd_allg[column]
            occupancy_time_g = occupancy_time_allg[column]
            rel_occupancy_time_g = rel_occupancy_time_allg[column -1] if g !=0 and g !=3 else None
            
            if g == 1:
                relhd_fulltrial = relhd_fulltrial_g1
            elif g == 2:
                relhd_fulltrial = relhd_fulltrial_g2
            
            # Load spike data for this goal in frames
            spike_train = get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, x_g)
        

            
            # ===== Plot ratemap ====
            rmap, x_edges, y_edges=  get_ratemaps_restrictedx(spike_train, x_fulltrial, y_fulltrial, x_g, y_g, 3, binsize=36, stddev=25)

            plot_rmap(rmap, xmin, xmax, ymin, ymax, x_edges, y_edges, outline_x, outline_y, ax = axs[0, column], fig = fig, title = f"g{g}, n = {len(spike_train)}")

            # === Plot HD ===
            axs[1, column].remove()
            axs[1, column] = fig.add_subplot(3, len(goals), len(goals) + column + 1, projection="polar")

            axs[2, column].remove()
            axs[2, column] = fig.add_subplot(3, len(goals), 2*len(goals) + column + 1, projection="polar")


            direction_firing_rate, bin_centers = get_directional_firingrate(hd_fulltrial, spike_train, num_bins, occupancy_time_g)
            
            #fig.delaxes(axs[1,column])
            #axs[1, column] = fig.add_subplot(2, len(goals), len(goals) + column + 1, polar=True)
            plot_directional_firingrate(bin_centers, direction_firing_rate, ax = axs[1,column])
            
            if g == 1 or g == 2:
                # === Plot relative HD ===
                direction_firing_rate_rel, bin_centers = get_directional_firingrate(relhd_fulltrial, spike_train, num_bins,  rel_occupancy_time_g)
                
                #fig.delaxes(axs[2,column])
                #axs[2, column] = fig.add_subplot(3, len(goals), len(goals) + column + 1, polar=True)

                plot_directional_firingrate(bin_centers, direction_firing_rate_rel, ax = axs[2,column])


        output_path = os.path.join(output_folder, f"unit_{unit_id}_rm_hd.png")
        if saveplots:
            plt.savefig(output_path)
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
            
        
    
    print(f"Saved plots to {output_folder}")
        
def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    """
    Copied from Pycircstat documentation
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r


def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # Copied from picircstat documentation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"

    return ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) /
            np.sum(w, axis=axis))       

if __name__ == "__main__":    
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    plot_ratemaps_and_hd_pergoal(derivatives_base,unit_type = "pyramidal", include_g0 = False, saveplots=True, show_plots=True)



