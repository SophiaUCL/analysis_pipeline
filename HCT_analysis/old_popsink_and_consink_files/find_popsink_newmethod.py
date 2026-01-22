import matplotlib
import numpy as np
import pandas as pd
from matplotlib.path import Path
import os
import spikeinterface.extractors as se
from utilities.platforms_utils import get_platform_center, calculate_occupancy_plats, get_hd_distr_allplats, get_firing_rate_platforms, get_norm_hd_distr
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from calculate_occupancy import get_direction_bins
from population_sink.get_relDirDist import calculate_relDirDist
from population_sink.calculate_MRLval import mrlData
from population_sink.plot_plat_info import plot_plat_info
from utilities.mrl_func import resultant_vector_length
from utilities.utils import get_unit_ids
from astropy.stats import circmean
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.trials_utils import get_limits_from_json, get_goal_numbers, get_coords
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from tqdm import tqdm
matplotlib.use("TkAgg")
import pickle
import datetime

def load_directories(derivatives_base):
    """ Loads directories that we need and pos data"""
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    kilosort_output_path = os.path.join(derivatives_base, "ephys", "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_w_platforms.csv')
    pos_data = pd.read_csv(pos_data_path)

    if np.nanmax(pos_data['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data['hd'] = np.deg2rad(pos_data['hd'])
    
    pos_data_path_intervals =  os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_allintervals.csv')
    pos_data_iv = pd.read_csv(pos_data_path_intervals)

    if np.nanmax(pos_data_iv['hd']) > 2* np.pi + 0.1: # Check if angles are in radians
        pos_data_iv['hd'] = np.deg2rad(pos_data_iv['hd'])
    
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'population_sink_newmethod')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    data_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'popsink_data_newmethod')
    return rawsession_folder, sorting, pos_data, pos_data_iv, output_folder, data_folder

def restrict_pos_data(derivatives_base, g, pos_data_iv):
    """ Returns positional data used for the analaysis for goal g
    if g < 3 (meaning g0, 1 or 2), we only load positional data for that goal
    if g == 3, we use the data for the full interval"""
    if g < 3:
        pos_data_path_g = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{g}_trials.csv')
        pos_data_g = pd.read_csv(pos_data_path_g)
        platform_occupancy_g = calculate_occupancy_plats(pos_data_g) # platform occupancy in frames
        hd_distr_g, bin_centers = get_hd_distr_allplats(pos_data_g)
    else:
        pos_data_g = pos_data_iv
        platform_occupancy_g = calculate_occupancy_plats(pos_data_iv)
        hd_distr_g, bin_centers = get_hd_distr_allplats(pos_data_iv)
    return pos_data_g, platform_occupancy_g, hd_distr_g, bin_centers

def get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, pos_data):
    """ Gets the spiketrain for unit unit_id for goal g. Spiketrain returned is in frames
    If g < 3 (so g0, 1 or 2), we restrict the spiketrain to only that goal"""
    
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_secs = np.round(spike_train_unscaled/sample_rate) # trial data is now in frames in order to match it with xy data
    
    # If we're only looking at one goal, restrict the spiketrain to match only period of goal 1 or 2
    if g < 3:
        spike_train_restricted = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder=rawsession_folder, goal=g)
    else:
        spike_train_restricted = spike_train_secs
    spike_train = spike_train_restricted*frame_rate
    spike_train = [np.int32(el) for el in spike_train if el < len(pos_data)]
    return spike_train

def get_norm_mean_angle(norm_hd_distr, bin_centers):
    """ Returns mean angle for each platform. If all values are nan or 0, returns nan"""
    norm_mean_angle = []
    for i in range(61):
        if np.all(np.isnan(norm_hd_distr[i])) or np.all(norm_hd_distr[i] == 0):
            norm_mean_angle.append(np.nan)
        else:
            norm_mean_angle.append(circmean(bin_centers, weights=norm_hd_distr[i]))
    return norm_mean_angle

def get_data_popsink_calc(hcoord, vcoord, mean_norm_vecs_angle, mean_norm_vecs_length):
    """ This data will be used for consink calculation"""
    pos_org = np.array([get_platform_center(hcoord, vcoord, p+1) for p in range(61)]) # positions are the center for each platform
    plats_org = np.arange(1,62)
    hd_org = mean_norm_vecs_angle # hd is the mean angle for each platform
    nspikes_org = np.round(np.array(mean_norm_vecs_length)*100).astype(int) 
    return pos_org, plats_org, hd_org, nspikes_org

def remove_nan(pos_org, plats_org, hd_org, nspikes_org, indices_to_del):
    """ Removes nans from the data used for consink calculation"""
    plats = np.delete(plats_org, indices_to_del)
    pos = np.delete(pos_org, indices_to_del, axis=0)
    hd = np.delete(hd_org, indices_to_del)
    nspikes = np.delete(nspikes_org, indices_to_del)
    if len(plats) != len(hd) or len(plats) != len(pos) or len(hd) != len(pos) or len(nspikes) != len(pos):
        raise ValueError("Lengths of plats, hd, and pos do not match after removing NaNs.")
    return pos, plats, hd, nspikes

def create_spikedata(pos, hd, plats, nspikes):
    """ Repeats each value by nspikes"""
    spikePos = np.repeat(pos, nspikes, axis = 0)
    spikeHD = np.repeat(hd, nspikes)
    spikePlats = np.repeat(plats, nspikes)
    return spikePos, spikeHD, spikePlats         

def save_session_data(data_folder, pos, hd, nspikes, pos_org, platform_occupancy_g, scaled_vecs_allplats, allscales, allfiring_rates, allnorm_MRL, hcoord, vcoord, unit_type, hd_org, nspikes_org, g, relDirDist):
    """ Saves session data"""
    session_data = {
        'pos': pos,
        'hd': hd,
        'nspikes': nspikes,
        'pos_org': pos_org,
        'hd_org': hd_org,
        'nspikes_org': nspikes_org,
        'occupancy': platform_occupancy_g,
        'scaled_vecs_allplats': scaled_vecs_allplats,
        'allscales': allscales,
        'allfiring_rates': allfiring_rates,
        'allnorm_MRL': allnorm_MRL,
        'hcoord': hcoord,
        'vcoord': vcoord,
        'timestamp': datetime.datetime.now().isoformat(),
        'goal': g,
        'unit_type': unit_type,
        'relDirDist': relDirDist,
        'comment': 'Intermediate population sink data for verification and reuse.'
    }

    # Save the dictionary as a pickle
    session_filename = os.path.join(data_folder, f'session_vars_goal{g}_{unit_type}.pkl')
    with open(session_filename, 'wb') as f:
        pickle.dump(session_data, f)

    print(f"✅ Saved session variables to {session_filename}")
             
             
                                     
def calculate_popsink(derivatives_base, unit_type, title = 'Population Sinks', run_zero = True, frame_rate = 25, sample_rate = 30000, code_to_run = []):
    """
    calculates the population sink for the whole trial, for units split into goal 1 and units split into goal 2
    
    NOTE: Again, reldirdist can be calculated two ways: per goal or for full trial. I have to try both methods!!!!!!
    """
    if unit_type not in ['pyramidal', 'good', 'all']:
        raise ValueError('unit type not correctly defined')

    # Load directories and data
    rawsession_folder, sorting, pos_data, pos_data_iv, output_folder, data_folder = load_directories(derivatives_base) 
    
    # Loading units
    unit_ids = sorting.unit_ids
    unit_ids = get_unit_ids(unit_ids, unit_type, derivatives_base)


    # To load data
    limits = get_limits_from_json(derivatives_base)
    sink_bins = load_pickle('sink_bins', data_folder)
    direction_bins = load_pickle('direction_bins', data_folder)
    goal_numbers = get_goal_numbers(derivatives_base)
    hcoord, vcoord = get_coords(derivatives_base)
    
    # Loop over all units
    if 1 in code_to_run:
        platform_occupancy_allgoals = []
        for g in range(4):
        # Here g == 3 corresponds to the whole session, not split into goals
            scaled_vecs_allplats = []
            allscales = []
            allfiring_rates = []
            allnorm_MRL =[]
            
            if g == 0 and not run_zero:
                continue
            pos_data_g, platform_occupancy_g, hd_distr_g, bin_centers= restrict_pos_data(derivatives_base, g, pos_data_iv) # NOTE: Here we use iv data for g ==3 

            # Find all platforms where occupancy is below 10 seconds
            p_low_occ = [np.where(np.array(platform_occupancy_g) < frame_rate*10)[0]]

            for val, unit_id in tqdm(enumerate(unit_ids)):
                # Load spike times
                spike_train = get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, pos_data)
                if len(spike_train) == 0:
                    continue
                # Get firing rate and normalised hd distr
                plat_firing_rate = get_firing_rate_platforms(spike_train, pos_data, platform_occupancy_g)
                norm_hd_distr = get_norm_hd_distr(spike_train, pos_data, hd_distr_g)

                # normalised MRL and mean angle
                norm_MRL = [resultant_vector_length(bin_centers, w=norm_hd_distr[i]) for i in range(61)]
                norm_mean_angle = get_norm_mean_angle(norm_hd_distr, bin_centers)

                # This is by how much we'll scale the unit vectors
                scale_factor = [norm_MRL[p] *plat_firing_rate[p] for p in range(61)]

                # Make unit vectors from mean angle and scale them by scale_vector
                scaled_vecs = [[np.cos(norm_mean_angle[p])*scale_factor[p], np.sin(norm_mean_angle[p])*scale_factor[p]] for p in range(61)]
                
                # Low occupancy set to nan
                scaled_vecs = np.array(scaled_vecs)
                scaled_vecs[p_low_occ] = np.nan  # Set low occupancy platforms to nan
                
                # Add data to variables
                allfiring_rates.append(plat_firing_rate)
                allnorm_MRL.append(norm_MRL)
                scaled_vecs_allplats.append( scaled_vecs)
                allscales.append(scale_factor)


            # Taking the mean for each platform
            mean_norm_vecs = np.nanmean(scaled_vecs_allplats, axis=0) 
            
            # Getting the length of each vec
            mean_norm_vecs_length = np.linalg.norm(mean_norm_vecs, axis=1)
            mean_norm_vecs_angle = np.arctan2(mean_norm_vecs[:,1], mean_norm_vecs[:,0])  # and angle 
            

            
            ##### CALCULATING CONSINK ##### 
            
            # 'spike data'
            pos_org, plats_org, hd_org, nspikes_org = get_data_popsink_calc(hcoord, vcoord, mean_norm_vecs_angle, mean_norm_vecs_length)

            # Nans for length and angle
            indices_to_del = np.where(np.isnan(mean_norm_vecs_length))
            indices_to_del = indices_to_del[0]
            
            pos, plats, hd, nspikes = remove_nan(pos_org, plats_org, hd_org, nspikes_org, indices_to_del) # Delete nans
            spikePos, spikeHD, spikePlats = create_spikedata(pos, hd, plats, nspikes) # Multiply each by nspikes
            
            # Calculation
            relDirDist = calculate_relDirDist(pos_data_g,  sink_bins, direction_bins)# Have to rewrite function for that
            mrl_dataset = mrlData(spikePos, spikeHD, spikePlats, relDirDist, direction_bins, sink_bins)
            
            # Saving
            if g == 3:
                name = "popsink_wholetrial"
            else:
                name = f"popsink_g{g}"
            
            print(f'Saved {name}')
            save_pickle(mrl_dataset, name,data_folder )
            plot_popsink_w_vectors(mrl_dataset, goal_numbers, hcoord, vcoord, limits, pos, hd, nspikes, output_folder, plot_name = name)
            #plot_plat_info(allfiring_rates, allnorm_MRL, allscales, hcoord, vcoord )

            # Gather all important variables from this run
            save_session_data(data_folder, pos, hd, nspikes, pos_org, platform_occupancy_g, scaled_vecs_allplats, allscales, allfiring_rates, allnorm_MRL, hcoord, vcoord, unit_type, hd_org, nspikes_org, g, relDirDist)
            platform_occupancy_allgoals.append(platform_occupancy_g)
    
        save_pickle(platform_occupancy_allgoals, 'platform_occupancy_allgoals', data_folder)
    
    if 2 in code_to_run:
        # Plotting
        wholetrial_data = load_pickle('popsink_wholetrial', data_folder)
        
        g1_data = load_pickle('popsink_g1', data_folder)
        g2_data = load_pickle('popsink_g2', data_folder)
        
        if run_zero:
            g0_data = load_pickle('popsink_g0', data_folder)
            mrls = [g0_data['mrl'], g1_data['mrl'], g2_data['mrl'],  wholetrial_data['mrl']]
            coords = [g0_data['coor'], g1_data['coor'], g2_data['coor'], wholetrial_data['coor']]
            angles = [g0_data['dir_deg'], g1_data['dir_deg'], g2_data['dir_deg'], wholetrial_data['dir_deg']]
        else:
            mrls = [g1_data['mrl'], g2_data['mrl'],  wholetrial_data['mrl']]
            coords = [g1_data['coor'], g2_data['coor'], wholetrial_data['coor']]
            angles = [g1_data['dir_deg'], g2_data['dir_deg'], wholetrial_data['dir_deg']]
        
        
        plot_popsink(mrls, coords, angles, goal_numbers, hcoord, vcoord,  limits, output_folder,run_zero = run_zero, plot_name=title)
        
        platform_occupancy_allgoals = load_pickle('platform_occupancy_allgoals', data_folder)
        plot_platform_occupancy(platform_occupancy_allgoals, hcoord, vcoord, limits, output_folder, run_zero = run_zero, plot_name = 'Platform Occupancy all goals')
        
        

def plot_platform_occupancy(platform_occupancy_allgoals, hcoord, vcoord, limits, output_folder, run_zero = True, plot_name = 'Platform Occupancy all goals', frame_rate = 25):
    """ Plots occupancy for all platforms for each goal """
    x_min, x_max, y_min, y_max = limits
    
    fig, axs = plt.subplots(1, 3 + run_zero, figsize=(8*(3 + run_zero), 8))
    axs = axs.flatten()
    
    cmap = plt.get_cmap('RdYlGn')
    
    for j in range(4): # j = 0, rat going to g2 during g1. j = 1, goal 1. j = 2, goal 2, j = 3 full trial
        if not run_zero and j == 0:
            continue
        ax = axs[j - (0 if run_zero else 1)]
        occupancy = platform_occupancy_allgoals[j - (0 if run_zero else 1)]
        occupancy = [el/frame_rate for el in occupancy]
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
        ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
        # plot the goal positions
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_max, y_min])
        ax.set_aspect('equal')
        
        # Add small text with MRL and angle on bottom
        if j == 3:
            title = 'Occupancy all trials'
        elif j != 0:
            title = f'Occupancy goal {j}'
        else:
            title = 'Occupancy going to G2 during G1'
        ax.set_title(title)
    plt.savefig(os.path.join(output_folder, f'{plot_name}.png'))
    print(f"Saved population sink plot to {output_folder}")
    plt.show()
    

def plot_popsink(mrls, coords, angles, goal_numbers, hcoord, vcoord,  limits, output_folder, run_zero = True,plot_name='Population Sinks'):
    """
    Plots popsink and the goal

    Args:
        data_folder (_type_): _description_
        popsink_coor (_type_): _description_
        goals (_type_): _description_
    """

    x_min, x_max, y_min, y_max = limits
    
    fig, axs = plt.subplots(1, 3 + run_zero, figsize=(18, 6))
    axs = axs.flatten()
    
    for j in range(4): # j = 0, rat going to g2 during g1. j = 1, goal 1. j = 2, goal 2, j = 3 full trial
        if not run_zero and j == 0:
            continue
        counter = j - (0 if run_zero else 1)
        ax = axs[counter]
        mrl = mrls[counter]
        popsink_coor = coords[counter]
        popsink_angle = angles[counter]
        
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            if j > 0 and j < 3 and i + 1 == goal_numbers[j -1]:
                colour = 'green'
            elif (j == 0 or j == 3) and i + 1 in goal_numbers:
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
        if j == 3:
            title = 'all trials'
        elif j != 0:
            title = f'goal {j}'
        else:
            title = 'Going to G2 during G1'
        ax.set_title(title)
    plt.savefig(os.path.join(output_folder, f'{plot_name}.png'))
    print(f"Saved population sink plot to {output_folder}")
    plt.show()

def plot_popsink_w_vectors(mrl_dataset, goal_numbers, hcoord, vcoord, limits, pos, hd, nspikes, output_folder, plot_name = 'Population sink with vector fields'):
    """
    Plots popsink and the goal (single section)
    Adds vectorfields

    """
    coor = mrl_dataset['coor']
    mrl = mrl_dataset['mrl']
    popsink_angle = mrl_dataset['dir_deg']
    x_min, x_max, y_min, y_max = limits


    x_pos = pos[:,0]
    y_pos = pos[:,1]
    
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        colour = 'grey'
        
        hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                            orientation=np.radians(28),  # Rotate hexagons to align with grid
                            facecolor=colour, alpha=0.2, edgecolor='k')
        ax.text(x, y, i + 1, ha='center', va='center', size=15)  # Start numbering from 1
        ax.add_patch(hex)

    ax.quiver(x_pos, y_pos, np.cos(hd)*nspikes, np.sin(hd)*nspikes)
    # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
    # plot the goal positions
    circle = plt.Circle((coor[0],coor[1]), 60, color='r', fill=True)
    ax.add_patch(circle)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_max, y_min])
    ax.set_aspect('equal')
    
    # Add small text with MRL and angle on bottom
    ax.text(700, 300, f'MRL: {mrl:.3f}, Angle: {popsink_angle:.1f}°', 
            ha='center', va='center')
    ax.set_title(plot_name)
    plt.savefig(os.path.join(output_folder, f'{plot_name}.png'))
    print(f"Saved population sink plot to {output_folder}")
    plt.show()          

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    #derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-003_id-2F\ses-01_date-17092025\all_trials"
    calculate_popsink(derivatives_base, unit_type = 'pyramidal', code_to_run = [0, 1, 2], title = 'Popsink Pyramida', run_zero = False, frame_rate = 25, sample_rate = 30000)
    