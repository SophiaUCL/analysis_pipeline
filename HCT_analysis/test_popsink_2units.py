import matplotlib
import numpy as np
import pandas as pd
from matplotlib.path import Path
import os
from calculate_pos_and_dir import get_directions_to_position, get_relative_directions_to_position
from calculate_occupancy import get_relative_direction_occupancy_by_position, get_axes_limits, get_direction_bins, \
    bin_directions, get_relative_direction_occupancy_by_position_platformbins
import spikeinterface.extractors as se
from utilities.platforms_utils import get_platform_center, calculate_occupancy_plats, get_hd_distr_allplats, \
    get_firing_rate_platforms, get_norm_hd_distr
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from find_consinks_main import find_consink_method2, find_consink, mean_resultant_length_nrdd
from utilities.mrl_func import resultant_vector_length
from utilities.utils import get_unit_ids
from astropy.stats import circmean
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.trials_utils import get_limits_from_json, get_goal_numbers, get_coords, get_sink_positions_platforms, get_coords_127sinks
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from tqdm import tqdm
matplotlib.use("TkAgg")
from find_popsink_main import restrict_pos_data, load_directories, create_spikedata, get_spike_train, get_norm_mean_angle, plot_platform_occupancy,plot_popsink_w_vectors, remove_nan, get_data_popsink_calc, get_dir_allframes, save_session_data,  get_dirdist_from_reldir_allframes, find_popsink_m3
""" This version plots just 2 units for one goal and calculates the population sink from them in order to test the code"""







def plot_unit_metrics(norm_mean_angle, metric, hcoord, vcoord, limits,
                           plot_name='Population sink with vector fields', ax=None):
    """
    Plots the metrics for a unit (firing rate, MRL, or firing rate * MRL)

    """
    x_min, x_max, y_min, y_max = limits

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        colour = 'grey'

        hex = RegularPolygon((x, y), numVertices=6, radius=87.,
                             orientation=np.radians(28),  # Rotate hexagons to align with grid
                             facecolor=colour, alpha=0.2, edgecolor='k')
        ax.text(x, y, i + 1, ha='center', va='center', size=10)  # Start numbering from 1
        ax.add_patch(hex)
    ax.quiver(hcoord, vcoord, np.cos(norm_mean_angle) * metric, np.sin(norm_mean_angle) * metric)
    # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c='grey')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_max, y_min])
    ax.set_aspect('equal')

    ax.set_title(plot_name)
    if ax is None:
        plt.show()


def export_unit_metrics_csv_wide(
    allfiring_rates,
    allnorm_MRL,
    allscales,
    allangles,
    mean_norm_vecs_length,
    mean_norm_vecs_angle,
    unit_ids,
    output_folder,
    filename="unit_metrics_by_platform_wide.csv"
):
    """
    Columns:
      unit_X_firing_rate, unit_X_MRL, unit_X_scale_factor,
      mean_firing_rate, mean_MRL, mean_scale_factor,
      mean_vector_length, mean_vector_angle_rad, mean_vector_angle_deg

    Rows:
      platforms
    """

    allfiring_rates = np.array(allfiring_rates)
    allnorm_MRL = np.array(allnorm_MRL)
    allscales = np.array(allscales)

    n_units, n_platforms = allfiring_rates.shape

    data = {
        "platform": np.arange(1, n_platforms + 1)
    }

    # Per-unit columns
    for u, unit_id in enumerate(unit_ids[:n_units]):
        data[f"unit_{unit_id}_firing_rate"] = allfiring_rates[u]
        data[f"unit_{unit_id}_MRL"] = allnorm_MRL[u]
        data[f"unit_{unit_id}_scale_factor"] = allscales[u]
        data[f"unit_{unit_id}_angle"] = allangles[u]

    # Mean across units (per platform)
    data["mean_firing_rate"] = np.nanmean(allfiring_rates, axis=0)
    data["mean_MRL"] = np.nanmean(allnorm_MRL, axis=0)
    data["mean_scale_factor"] = np.nanmean(allscales, axis=0)

    # Population vector metrics
    data["mean_vector_length"] = mean_norm_vecs_length
    data["mean_vector_angle_rad"] = mean_norm_vecs_angle
    data["mean_vector_angle_deg"] = np.rad2deg(mean_norm_vecs_angle)

    df = pd.DataFrame(data)

    outpath = os.path.join(output_folder, filename)
    df.to_csv(outpath, index=False)



def calculate_popsink_fewunits(derivatives_base, unit_type, num_units = 2, g = 1,  title='Population Sinks',frame_rate=25,
                      sample_rate=30000, code_to_run=[]):
    """
    calculates the population sink for the whole trial, for units split into goal 1 and units split into goal 2

    NOTE: Again, reldirdist can be calculated two ways: per goal or for full trial. I have to try both methods!!!!!!
    """
    if unit_type not in ['pyramidal', 'good', 'all']:
        raise ValueError('unit type not correctly defined')

    # Load directories and data
    rawsession_folder, sorting, pos_data, pos_data_iv, output_folder= load_directories(derivatives_base, dirname = 'popsink_fewunits')

    # Loading units
    unit_ids = sorting.unit_ids
    unit_ids = get_unit_ids(unit_ids, unit_type, derivatives_base)

    # To load data
    limits = get_limits_from_json(derivatives_base)
    direction_bins = get_direction_bins()
    goal_numbers = get_goal_numbers(derivatives_base)
    hcoord, vcoord = get_coords(derivatives_base)
    sink_positions = get_sink_positions_platforms(derivatives_base)
    hcoord_127, vcoord_127 = get_coords_127sinks(derivatives_base)
    methods = ['trial_norm','plat_norm', 'no_norm']

    ncols = 3
    nrows = num_units + 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*8, nrows*8))
    fig.suptitle(title)

    unit_ids = unit_ids[1:num_units+1]
    # Loop over all units
    if 1 in code_to_run:
        platform_occupancy_allgoals = []

        # Here g == 3 corresponds to the whole session, not split into goals
        scaled_vecs_allplats = []
        allscales = []
        allfiring_rates = []
        allnorm_MRL = []
        allangles = []

        # For each goal, obtains its positional data, and for each platform,
        # calculat the head direction distribution and occupancy
        pos_data_g, platform_occupancy_g, hd_distr_g, bin_centers = restrict_pos_data(derivatives_base, g,
                                                                                      pos_data_iv)  # NOTE: Here we use iv data for g ==3

        # Find all platforms where occupancy is below 10 seconds
        p_low_occ = [np.where(np.array(platform_occupancy_g) < frame_rate * 10)[0]]

        for val, unit_id in tqdm(enumerate(unit_ids)):

            # Load spike times
            spike_train = get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, pos_data)
            if len(spike_train) == 0:
                continue
            # Get firing rate and normalised hd distr
            plat_firing_rate = get_firing_rate_platforms(spike_train, pos_data, platform_occupancy_g)
            norm_hd_distr = get_norm_hd_distr(spike_train, pos_data, hd_distr_g)

            # normalised MRL and mean angle
            norm_MRL_org = [resultant_vector_length(bin_centers, w=norm_hd_distr[i]) for i in range(61)]
            norm_mean_angle = get_norm_mean_angle(norm_hd_distr, bin_centers)

            norm_MRL =np.array(norm_MRL_org)
            norm_MRL[p_low_occ[0]] = 0

            # This is by how much we'll scale the unit vectors
            scale_factor = [norm_MRL[p] * plat_firing_rate[p] for p in range(61)]

            # Make unit vectors from mean angle and scale them by scale_vector
            scaled_vecs = [
                [np.cos(norm_mean_angle[p]) * scale_factor[p], np.sin(norm_mean_angle[p]) * scale_factor[p]] for p
                in range(61)]

            # Low occupancy set to nan
            scaled_vecs = np.array(scaled_vecs)
            scaled_vecs[p_low_occ] = np.nan  # Set low occupancy platforms to nan

            # Add data to variables
            allfiring_rates.append(plat_firing_rate)
            allnorm_MRL.append(norm_MRL)
            scaled_vecs_allplats.append(scaled_vecs)
            allscales.append(scale_factor)
            allangles.append(np.rad2deg(norm_mean_angle))

            names = ['firing rate', 'MRL', 'Firing rate *MRL']
            for i, metric, name in zip([0,1,2], [plat_firing_rate, norm_MRL, scale_factor], names):
                plot_unit_metrics(norm_mean_angle, metric, hcoord, vcoord, limits, plot_name=name, ax=axs[val, i])


        # Taking the mean for each platform
        mean_norm_vecs = np.nanmean(scaled_vecs_allplats, axis=0)

        # Getting the length of each vec
        mean_norm_vecs_length = np.linalg.norm(mean_norm_vecs, axis=1)
        mean_norm_vecs_angle = np.arctan2(mean_norm_vecs[:, 1], mean_norm_vecs[:, 0])  # and angle

        breakpoint()

        plot_unit_metrics(mean_norm_vecs_angle, mean_norm_vecs_length, hcoord, vcoord, limits, plot_name='norm length', ax=axs[2,2])
        export_unit_metrics_csv_wide(
            allfiring_rates=allfiring_rates,
            allnorm_MRL=allnorm_MRL,
            allscales=allscales,
            allangles = allangles,
            mean_norm_vecs_length=mean_norm_vecs_length,
            mean_norm_vecs_angle=mean_norm_vecs_angle,
            unit_ids=unit_ids,
            output_folder=output_folder
        )
        # 'spike data'
        pos_org, plats_org, hd_org, nspikes_org = get_data_popsink_calc(hcoord, vcoord, mean_norm_vecs_angle,
                                                                        mean_norm_vecs_length)


        # Nans for length and angle
        indices_to_del = np.where(np.isnan(mean_norm_vecs_length))
        indices_to_del = indices_to_del[0]

        pos, plats, hd, nspikes = remove_nan(pos_org, plats_org, hd_org, nspikes_org, indices_to_del)  # Delete nans
        spikePos, spikeHD, spikePlats = create_spikedata(pos, hd, plats, nspikes)  # Multiply each by nspikes
        fake_spike_train = np.arange(len(spikePos), dtype=int)
        # Calculation
        ## THESE TWO I REWRITE
        sinkdir_allframes, reldir_allframes = get_dir_allframes(spikePos, spikeHD, sink_positions)
        reldir_occ_by_pos = get_relative_direction_occupancy_by_position_platformbins(pos_data_g, sink_positions, num_candidate_sinks= 127, n_dir_bins=12, frame_rate=25)


        for m, method in enumerate(methods):
            if method == 'trial_norm':
                max_mrl, max_mrl_indices, mean_angle = find_consink(
                    fake_spike_train, reldir_occ_by_pos, direction_bins, pos_data,
                    reldir_allframes, platforms_spk = spikePlats, verify_nans = False)
            elif method == "plat_norm":
                max_mrl, max_mrl_indices, mean_angle = find_consink_method2(
                    fake_spike_train, reldir_occ_by_pos, direction_bins, pos_data,
                    reldir_allframes, platforms_spk=spikePlats, verify_nans=False)
            else:
                max_mrl, max_mrl_indices, mean_angle = find_popsink_m3(reldir_allframes, direction_bins)
            # Saving
            if g == 3:
                name = f"popsink_wholetrial_{method}"
                title = f"G1 + G2, {method}"
            else:
                name = f"popsink_g{g}_{method}"
                title = f"Goal {g}, {method}"

            print(f'Saved {name}')

            consink_plat = max_mrl_indices[0][0] + 1

            mrl_dataset = {'mrl': max_mrl,
                'coor': [hcoord_127[consink_plat - 1], vcoord_127[consink_plat - 1]],
                'platform': consink_plat,
                'dir': mean_angle,
                'dir_deg': np.rad2deg(mean_angle),
                'pos': pos,
                'hd': hd,
                'nspikes': nspikes}

            save_pickle(mrl_dataset, name, output_folder)
            #plot_popsink_w_vectors(mrl_dataset, goal_numbers, hcoord, vcoord, limits, output_folder,
                                   #plot_name=title, ax = axs[m, g - 1 + run_zero])
            # plot_plat_info(allfiring_rates, allnorm_MRL, allscales, hcoord, vcoord )

            # Gather all important variables from this runS
            save_session_data(output_folder, method, pos, hd, nspikes, pos_org, platform_occupancy_g, scaled_vecs_allplats,
                              allscales, allfiring_rates, allnorm_MRL, hcoord, vcoord, unit_type, hd_org, nspikes_org,
                              g, reldir_occ_by_pos)
            plot_popsink_w_vectors(mrl_dataset, hcoord, vcoord, limits, output_folder, plot_name=f'Method {m}',
                                   ax=axs[3, m])
        plt.savefig(os.path.join(output_folder, "overview.png"))
        plt.show()
        platform_occupancy_allgoals.append(platform_occupancy_g)
        save_pickle(platform_occupancy_allgoals, 'platform_occupancy_allgoals', output_folder)

        print(f"Saved population sink plot to {output_folder}")

        platform_occupancy_allgoals = load_pickle('platform_occupancy_allgoals', output_folder)
        #plot_platform_occupancy(platform_occupancy_allgoals, hcoord, vcoord, limits, output_folder, run_zero=run_zero,
                                #plot_name='Platform Occupancy all goals')



if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    # derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-003_id-2F\ses-01_date-17092025\all_trials"
    calculate_popsink_fewunits(derivatives_base, unit_type='pyramidal', code_to_run=[1], frame_rate=25, sample_rate=30000)
