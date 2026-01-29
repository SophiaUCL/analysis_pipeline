import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
import pandas as pd
import sys

from HCT_analysis.utilities.load_and_save_data import load_pickle, save_pickle
from HCT_analysis.utilities.trials_utils import get_goal_coordinates
from HCT_analysis.utilities.mrl_func import resultant_vector_length

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






def plot_vector_fields_all(
    unit_ids,
    vector_fields,
    goal_coordinates,
    x_centres,
    y_centres,
    plot_dir,
    goals_to_include,
    methods,
    output_folder,
):
    os.makedirs(plot_dir, exist_ok=True)

    for u in unit_ids:
        fig, axes = plt.subplots(
            1, len(goals_to_include),
            figsize=(6 * len(goals_to_include), 6),
            squeeze=False
        )
        axes = axes[0]
        fig.suptitle(f"Vector fields – unit {u}", fontsize=20)

        for i, g in enumerate(goals_to_include):
            ax = axes[i]

            # ---------- title ----------
            if g == 0:
                ax.set_title("G1 → G2")
            else:
                ax.set_title(f"Goal {g}")

            # ---------- goal circles ----------
            if g > 0:
                gx, gy = goal_coordinates[g - 1]
                ax.add_patch(plt.Circle((gx, gy), 80, fill=False, color='green', lw=3))
            else:
                for gx, gy in goal_coordinates:
                    ax.add_patch(plt.Circle((gx, gy), 80, fill=False, color='green', lw=3))

            # ---------- vector field ----------
            try:
                vf = vector_fields[g]['units'][int(u)]
                ax.quiver(
                    x_centres,
                    y_centres,
                    np.cos(vf),
                    np.sin(vf),
                    color='black',
                    scale=10
                )
            except KeyError:
                continue

            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.tick_params(labelsize=10)

            # ---------- consinks (all methods) ----------
            if g > 0:
                y_text = 0.95
                y_step = 0.08

                for k, m in enumerate(methods):
                    consinks_df = load_pickle(f'consinks_df_m{m}', output_folder)
                    row = consinks_df.loc[u]

                    mrl = row[f'mrl_g{g}']
                    ci95 = row[f'ci_95_g{g}']
                    pos = row[f'position_g{g}']
                    ang = row[f'mean_angle_g{g}']

                    if not np.isfinite(mrl) or pos is None:
                        continue

                    if ang > np.pi:
                        ang -= 2 * np.pi

                    is_sig = np.isfinite(ci95) and mrl > ci95
                    color = 'red' if is_sig else 'grey'

                    style = METHOD_STYLE[m]

                    # ---- marker ----
                    ax.scatter(
                        pos[0],
                        pos[1],
                        marker=style['marker'],
                        s=260,
                        facecolors=color,
                        edgecolors='black',
                        linewidths=1.2,
                        zorder=6,
                        label=style['label'] if i == 0 else None
                    )

                    # ---- text ----
                    ax.text(
                        0.02,
                        y_text - k * y_step,
                        f"M{m}: mrl={mrl:.2f}, ci95={ci95:.2f}, θ={np.rad2deg(ang):.1f}°",
                        transform=ax.transAxes,
                        fontsize=11,
                        color=color,
                        va='top'
                    )

            if i == 0:
                ax.legend(frameon=False, loc='lower left')

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(plot_dir, f"vector_fields_unit_{u}.png"), dpi=300)
        plt.close(fig)

        
def main(derivatives_base, methods = [1,2,3], goals_to_include = [0,1,2]):
    """
    For each unit, creates a 3x1 subplot of vector fields for all trials, goal 1 trials, and goal 2 trials.
    run get_directional_occupancy_by_pos.py first to generate the necessary data.

    Args:
        derivatives_base (str): The base directory for the derivatives.
        
    Note! Goal coordinates is not correctly defined yet
    """
    rawsession_folder= derivatives_base.replace("\derivatives", "\rawdata")
    rawsession_folder =os.path.dirname(rawsession_folder)
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


    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics',  'spatial_features', 'consinks')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    goal_coordinates = get_goal_coordinates(derivatives_base, rawsession_folder)
    
    vector_fields = {}
    mean_resultant_lengths = {}
        

    ############# CALCULATING VECTOR FIELDS ##############3
    

    # Now per goal
    for g in goals_to_include:# AGAIN, We're not doing full trial here
        print("Calculating vector fields for goal ", g)
        spike_rates_by_position_and_direction_by_goal = load_pickle( f'spike_rates_by_position_and_direction_g{g}', output_folder)
        vector_fields[g], mean_resultant_lengths[g] = calculate_vector_fields(spike_rates_by_position_and_direction_by_goal)
    print("Saving")
    # Save
    save_pickle(vector_fields, 'vector_fields', output_folder)
    save_pickle(mean_resultant_lengths, 'mean_resultant_lengths', output_folder)
    
    # Plotting
    plot_dir = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'vector_fields')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    print(f"Saving to {plot_dir}")
    x_bins = vector_fields[g]['x_bins']
    x_centres = x_bins[:-1] + np.diff(x_bins)/2
    y_bins = vector_fields[g]['y_bins']
    y_centres = y_bins[:-1] + np.diff(y_bins)/2
    
    plot_vector_fields_all(unit_ids,vector_fields,goal_coordinates,x_centres,y_centres,plot_dir,goals_to_include,methods,output_folder)

    



if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    main(derivatives_base, rawsession_folder)










    