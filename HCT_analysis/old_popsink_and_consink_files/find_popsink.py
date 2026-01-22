import matplotlib
import numpy as np
import pandas as pd
from matplotlib.path import Path
import os
import spikeinterface.extractors as se
from utilities.platforms_utils import get_platform_center, calculate_occupancy_plats, get_hd_distr_allplats, get_firing_rate_platforms, get_norm_hd_distr
from utilities.restrict_spiketrain import restrict_spiketrain
from calculate_occupancy import get_direction_bins
from population_sink.get_relDirDist import calculate_relDirDist
from population_sink.calculate_MRLval import mrlData, getRelDirDist, mrlRelDir
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from utilities.mrl_func import resultant_vector_length
from astropy.stats import circmean
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.trials_utils import get_limits_from_json, get_goal_numbers, get_coords
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from tqdm import tqdm
matplotlib.use("TkAgg")

def restrict_pos_data(derivatives_base, g, pos_data_iv):
    """ Returns positional data used for the analaysis for goal g
    if g < 3 (meaning g0, 1 or 2), we only load positional data for t
    if g == 3, we use the data for the full interval"""
    if g < 3:
        pos_data_path_g = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                       f'XY_HD_goal{g}_trials.csv')
        pos_data_g = pd.read_csv(pos_data_path_g)
        platform_occupancy_g = calculate_occupancy_plats(pos_data_g)
        hd_distr_g, bin_centers = get_hd_distr_allplats(pos_data_g)
    else:
        pos_data_g = pos_data_iv
        platform_occupancy_g = calculate_occupancy_plats(pos_data_iv)
        hd_distr_g, bin_centers = get_hd_distr_allplats(pos_data_iv)
    return pos_data_g, platform_occupancy_g, hd_distr_g, bin_centers

def calculate_popsink(derivatives_base,  unit_type, title = 'Population Sink', frame_rate = 25, sample_rate = 30000, code_to_run = []):
    """
    calculates the population sink for the whole trial, for units split into goal 1 and units split into goal 2
    
    NOTE: Again, reldirdist can be calulcated two ways: per goal or for full trial. I have to try both methods!!!!!!
    """
    if unit_type not in ['pyramidal', 'good', 'all']:
        raise ValueError('unit type not correctly defined')
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    goal_numbers = get_goal_numbers(derivatives_base)
    
    hcoord, vcoord = get_coords(derivatives_base)
    
    # Loading spike data
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    pos_data_path_intervals = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD',
                                           'XY_HD_allintervals.csv')
    pos_data_iv = pd.read_csv(pos_data_path_intervals)

    if np.nanmax(pos_data_iv['hd']) > 2 * np.pi + 0.1:  # Check if angles are in radians
        pos_data_iv['hd'] = np.deg2rad(pos_data_iv['hd'])

    if unit_type == 'good':

        good_units_path = os.path.join(derivatives_base, "ephys", "concat_run", "sorting","sorter_output", "cluster_group.tsv")
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
        # Loading pyramidal units
    elif unit_type == 'pyramidal':
        pyramidal_units_path = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features","all_units_overview", "pyramidal_units_2D.csv")
        print("Getting pyramidal units 2D")
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units

    # Loading xy data
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_w_platforms.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])
        
    # Output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'population_sink')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # To load data
    data_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'consink_data')
    limits = get_limits_from_json(derivatives_base)
    fig, axs = plt.subplots(4,3, figsize = (18,24))
    fig.suptitle(title, fontsize=20)
    # Loop over all units
    if 1 in code_to_run:
        platform_occupancy_allgoals = []
        n_spikes_platforms_allgoals = []

        for g in [1,2,3]:
        # Here g == 0 corresponds to the whole session, not split into goals
            scaled_vecs_allplats = []
            n_spikes_per_platform = np.zeros(61, dtype=int)

            # calculat the head direction distribution and occupancy
            pos_data_g, platform_occupancy_g, hd_distr_g, bin_centers = restrict_pos_data(derivatives_base, g,
                                                                    pos_data_iv)
            p_low_occ = [np.where(np.array(platform_occupancy_g) < frame_rate * 10)[0]]
            for unit_id in tqdm(unit_ids):
                # Load spike times
                spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
                spike_train_secs = np.round(spike_train_unscaled/sample_rate) # trial data is now in frames in order to match it with xy data
                
                # If we're only looking at one goal, restrict the spiketrain to match only period of goal 1 or 2
                if g < 3:
                    spike_train_restricted = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder=rawsession_folder, goal=g)
                else:
                    spike_train_restricted = spike_train_secs
                spike_train = spike_train_restricted*frame_rate
                spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]

                # Get firing rate
                #plat_firing_rate = get_firing_rate_platforms(spike_train, pos_data, platform_occupancy)

                platforms = pos_data['platform']
                platforms_spk = platforms[spike_train].to_numpy()
                platforms_spk = platforms_spk[~np.isnan(platforms_spk)]

                for p in platforms_spk:
                    n_spikes_per_platform[np.int32(p) - 1] += 1

                # Normalised hd distr
                #norm_hd_distr = get_norm_hd_distr(spike_train, pos_data, hd_distr_allplats)

                plat_firing_rate = get_firing_rate_platforms(spike_train, pos_data, platform_occupancy_g)
                norm_hd_distr = get_norm_hd_distr(spike_train, pos_data, hd_distr_g)
                # normalised MRL and mean angle
                norm_MRL = [resultant_vector_length(bin_centers, w=norm_hd_distr[i]) for i in range(61)]
                norm_mean_angle = []
                for i in range(61):
                    if np.all(np.isnan(norm_hd_distr[i])) or np.all(norm_hd_distr[i] == 0):
                        norm_mean_angle.append(np.nan)
                    else:
                        norm_mean_angle.append(circmean(bin_centers, weights=norm_hd_distr[i]))

                # This is by how much we'll scale the unit vectors
                scale_factor = [norm_MRL[p] *plat_firing_rate[p] for p in range(61)]

                # Make unit vectors from mean angle
                scaled_vecs = [[np.cos(norm_mean_angle[p])*scale_factor[p], np.sin(norm_mean_angle[p])*scale_factor[p]] for p in range(61)]

                scaled_vecs = np.array(scaled_vecs)
                scaled_vecs[p_low_occ] = np.nan  # Set low occupancy platforms to nan

                scaled_vecs_allplats.append(scaled_vecs)
            
            scaled_vecs_allplats = np.array(scaled_vecs_allplats) 
            # Taking the mean for each platform
            mean_norm_vecs = np.nanmean(scaled_vecs_allplats, axis=0) 
            
            # Getting the length of each vec
            mean_norm_vecs_length = np.linalg.norm(mean_norm_vecs, axis=1)
            mean_norm_vecs_angle = np.arctan2(-mean_norm_vecs[:,1], mean_norm_vecs[:,0])

            
            # Nans for length and angle
            indices_to_del = np.where(np.isnan(mean_norm_vecs_length))
            indices_to_del = indices_to_del[0]
            
            ##### CALCULATING CONSINK ##### 
            
            # We replace this with sinkbins
            sink_bins = load_pickle('sink_bins', data_folder)
            direction_bins = load_pickle('direction_bins', data_folder) # Should be the same as angle edges
            
            # 'spike data'
            pos = np.array([get_platform_center(hcoord, vcoord, p+1) for p in range(61)]) # positions are the center for each platform
            plats = np.arange(1,62)
            hd = mean_norm_vecs_angle # hd is the mean angle for each platform
            nspikes = np.round(np.array(mean_norm_vecs_length)*100).astype(int) 
            
            # Delelete nans
            plats = np.delete(plats, indices_to_del)
            pos = np.delete(pos, indices_to_del, axis=0)
            hd = np.delete(hd, indices_to_del)
            nspikes = np.delete(nspikes, indices_to_del)
            if len(plats) != len(hd) or len(plats) != len(pos) or len(hd) != len(pos) or len(nspikes) != len(pos):
                raise ValueError("Lengths of plats, hd, and pos do not match after removing NaNs.")
            
            spikePos = np.repeat(pos, nspikes, axis = 0)
            spikeHD = np.repeat(hd, nspikes)
            #spikePlats = np.repeat(plats, nspikes)
            relDirDist= calculate_relDirDist(pos_data,  sink_bins, direction_bins)# Have to rewrite function for that
            print(n_spikes_per_platform)

            n_spikes_per_platform[indices_to_del] = 0
            for m in range(2):
                if m == 0:
                    mrl_dataset= mrlData(spikePos, spikeHD, n_spikes_per_platform, relDirDist, direction_bins, sink_bins)
                    plot_name = f'Goal {g}, Jakes normalisation' if g < 3 else f'G1 + G2, Jakes normalisation'
                    mrl_dataset['hd'] = hd
                    mrl_dataset['nspikes'] = nspikes
                    mrl_dataset['pos'] = pos
                    plot_popsink_w_vectors(mrl_dataset, hcoord, vcoord, limits, output_folder, plot_name=plot_name,
                                           ax=axs[m, g - 1])
                else:
                    binEdges = direction_bins
                    dir_bin_centres = (binEdges[1:] + binEdges[:-1]) / 2
                    xAxis = sink_bins['x']
                    yAxis = sink_bins['y']
                    plot_name = f'Goal {g}, no normalisation' if g < 3 else f'G1 + G2, no normalisation'
                    dirRel2Goal_histCounts = getRelDirDist(spikePos, spikeHD, xAxis, yAxis, binEdges, normalize=False)
                    mrl_dataset = mrlRelDir(dirRel2Goal_histCounts, xAxis, yAxis, dir_bin_centres)  # should be bin centers?
                    mrl_dataset['hd'] = hd
                    mrl_dataset['nspikes'] = nspikes
                    mrl_dataset['pos'] = pos
                    plot_popsink_w_vectors(mrl_dataset, hcoord, vcoord, limits, output_folder, plot_name=plot_name,
                                           ax=axs[m, g - 1])
            # Saving
            if g == 0:
                name = "popsink_wholetrial"
            else:
                name = f"popsink_g{g}"
            mrl_dataset['hd'] = hd
            mrl_dataset['nspikes'] = nspikes
            mrl_dataset['pos'] = pos
            save_pickle(mrl_dataset, name,data_folder )

            platform_occupancy_g = np.array(platform_occupancy_g)
            platform_occupancy_g = platform_occupancy_g/frame_rate
            n_spikes_per_platform = np.array(n_spikes_per_platform)
            n_spikes_per_platform = n_spikes_per_platform/1000
            plot_platform_info(platform_occupancy_g, hcoord, vcoord, limits, plot_name = f'Occupancy {g}', ax = axs[2, g - 1])
            plot_platform_info(n_spikes_per_platform, hcoord, vcoord, limits, plot_name=f'Total spikes per platform goal {g} (x1000)', ax=axs[3, g - 1])
            platform_occupancy_allgoals.append(platform_occupancy_g)
            n_spikes_platforms_allgoals.append(n_spikes_per_platform)
            mean_norm_vecs_length[indices_to_del] = 0


        plt.savefig(os.path.join(output_folder, "Population sink overview"))
        print(f"Saved figure to {os.path.join(output_folder, "Population sink overview")}")
        plt.show()
        save_pickle(platform_occupancy_allgoals, 'platform_occupancy_allgoals', output_folder)
        save_pickle(n_spikes_platforms_allgoals, 'n_spikes_platforms_allgoals', output_folder)

    if 2 in code_to_run:
        # Plotting
        wholetrial_data = load_pickle('popsink_wholetrial', data_folder)
        g1_data = load_pickle('popsink_g1', data_folder)
        g2_data = load_pickle('popsink_g2', data_folder)
        
        mrls = [wholetrial_data['mrl'], g1_data['mrl'], g2_data['mrl']]
        coords = [wholetrial_data['coor'], g1_data['coor'], g2_data['coor']]
        angles = [wholetrial_data['dir_deg'], g1_data['dir_deg'], g2_data['dir_deg']]
        
        hcoord, vcoord = get_coords(derivatives_base)
        limits = get_limits_from_json(derivatives_base)
        plot_popsink(mrls, coords, angles, goal_numbers, hcoord, vcoord,  limits, output_folder, plot_name=title)
        platform_occupancy_allgoals = load_pickle('platform_occupancy_allgoals', output_folder)
        n_spikes_platforms_allgoals = load_pickle('n_spikes_platforms_allgoals', output_folder)
        plot_platform_occupancy(platform_occupancy_allgoals, hcoord, vcoord, limits, output_folder, run_zero=False,
                                plot_name='Platform Occupancy all goals')
        plot_platform_occupancy(n_spikes_platforms_allgoals, hcoord, vcoord, limits, output_folder, run_zero=False,
                                plot_name='Number of spikes per platform')

def plot_platform_occupancy(platform_occupancy_allgoals, hcoord, vcoord, limits, output_folder, run_zero=True,
                            plot_name='Platform Occupancy all goals', frame_rate=25):
    """ Plots occupancy for all platforms for each goal """
    x_min, x_max, y_min, y_max = limits

    fig, axs = plt.subplots(1, 3 + run_zero, figsize=(8 * (3 + run_zero), 8))
    axs = axs.flatten()

    cmap = plt.get_cmap('RdYlGn')

    for j in range(4):  # j = 0, rat going to g2 during g1. j = 1, goal 1. j = 2, goal 2, j = 3 full trial
        if not run_zero and j == 0:
            continue

        ax = axs[j - (0 if run_zero else 1)]
        occupancy = platform_occupancy_allgoals[j - (0 if run_zero else 1)]
        occupancy_normalized = occupancy / np.nanmax(occupancy)
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            if occupancy_normalized[i] == 0:
                colour = 'grey'
                text = ''
            else:
                colour = cmap(occupancy_normalized[i])
                text = f'{np.int32(occupancy[i])}'

            hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                                 orientation=np.radians(28),  # Rotate hexagons to align with grid
                                 facecolor=colour, alpha=0.2, edgecolor='k')
            ax.text(x, y, text, ha='center', va='center', size=15)  # Start numbering from 1
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c='grey')
        # plot the goal positions
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_max, y_min])
        ax.set_aspect('equal')

        # Add small text with MRL and angle on bottom
        if j == 3:
            title = 'Occupancy all trials (s)'
        elif j != 0:
            title = f'Occupancy goal {j} (s)'
        else:
            title = 'Occupancy going to G2 during G1 (s)'
        ax.set_title(title)
    plt.savefig(os.path.join(output_folder, f'{plot_name}.png'))
    plt.show()


def plot_platform_info(plat_info, hcoord, vcoord, limits,
                            plot_name, ax):
    """ Plots occupancy for all platforms for each goal """
    x_min, x_max, y_min, y_max = limits

    cmap = plt.get_cmap('RdYlGn')

    plat_info_normalized = plat_info / np.nanmax(plat_info)
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if plat_info[i] == 0:
            colour = 'grey'
            text = ''
        else:
            colour = cmap(plat_info_normalized [i])
            text = f'{np.int32(plat_info[i])}'

        hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                             orientation=np.radians(28),  # Rotate hexagons to align with grid
                             facecolor=colour, alpha=0.2, edgecolor='k')
        ax.text(x, y, text, ha='center', va='center', size=10)  # Start numbering from 1
        ax.add_patch(hex)

    # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c='grey')
    # plot the goal positions
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_max, y_min])
    ax.set_title(plot_name)
    ax.set_aspect('equal')


def plot_popsink_w_vectors(mrl_dataset, hcoord, vcoord, limits,output_folder,
                           plot_name='Population sink with vector fields', ax=None):
    """
    Plots popsink and the goal (single section)
    Adds vectorfields

    """

    pos = mrl_dataset['pos']
    hd = mrl_dataset['hd']
    nspikes = mrl_dataset['nspikes']
    coor = mrl_dataset['coor']
    mrl = mrl_dataset['mrl']
    popsink_angle = mrl_dataset['dir_deg']
    x_min, x_max, y_min, y_max = limits

    x_pos = pos[:, 0]
    y_pos = pos[:, 1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        colour = 'grey'

        hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                             orientation=np.radians(28),  # Rotate hexagons to align with grid
                             facecolor=colour, alpha=0.2, edgecolor='k')
        ax.text(x, y, i + 1, ha='center', va='center', size=5)  # Start numbering from 1
        ax.add_patch(hex)

    ax.quiver(x_pos, y_pos, np.cos(hd) * nspikes, np.sin(hd) * nspikes)
    # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c='grey')
    # plot the goal positions
    circle = plt.Circle((coor[0], coor[1]), 60, color='r', fill=True)
    ax.add_patch(circle)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_max, y_min])
    ax.set_aspect('equal')

    # Add small text with MRL and angle on bottom
    ax.text(x_min + 50, y_min + 100, f'MRL: {mrl:.3f}')
    ax.text(x_min + 50, y_min + 200, f'Angle: {popsink_angle:.1f}°')
    ax.set_title(plot_name)
    if ax is None:
        plt.savefig(os.path.join(output_folder, f'{plot_name}.png'))
        plt.show()


def plot_popsink(mrls, coords, angles, goal_numbers, hcoord, vcoord,  limits, output_folder, plot_name='Population Sinks'):
    """
    Plots popsink and the goal

    Args:
        data_folder (_type_): _description_
        popsink_coor (_type_): _description_
        goals (_type_): _description_
    """

    x_min, x_max, y_min, y_max = limits
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.flatten()
    
    for j in range(3): # j = 0, full trial. j = 1, goal 1. j = 2, goal 2
        ax = axs[j]
        mrl = mrls[j]
        popsink_coor = coords[j]
        popsink_angle = angles[j]
        
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            if j > 0 and i + 1 == goal_numbers[j -1]:
                colour = 'green'
            elif j == 0 and i + 1 in goal_numbers:
                colour = 'green'
            else:
                colour = 'grey'
            hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                                orientation=np.radians(28),  # Rotate hexagons to align with grid
                                facecolor=colour, alpha=0.2, edgecolor='k')
            ax.text(x, y, i + 1, ha='center', va='center', size=15)  # Start numbering from 1
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
        # plot the goal positions
        circle = plt.Circle((popsink_coor[0], popsink_coor[1]), 60, color='r', fill=True)
        ax.add_patch(circle)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_max, y_min])
        ax.set_aspect('equal')
        
        # Add small text with MRL and angle on bottom
        ax.text(700, 300, f'MRL: {mrl:.3f}, Angle: {popsink_angle:.1f}°', 
                ha='center', va='center')
        if j == 0:
            title = 'all trials'
        else:
            title = f'goal {j}'
        ax.set_title(title)
    plt.savefig(os.path.join(output_folder, f'{plot_name}.png'))
    print(f"Saved population sink plot to {output_folder}")
    plt.show()
                

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    calculate_popsink(derivatives_base, unit_type = 'pyramidal', title = 'Pyramidal popsink sub 1R ses 2 11/09', code_to_run = [1], frame_rate = 25, sample_rate = 30000)
    