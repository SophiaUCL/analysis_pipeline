import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import circmean
from HCT_analysis.utilities.mrl_func import resultant_vector_length
from HCT_analysis.utilities.trials_utils import  get_coords_127sinks
from HCT_analysis.utilities.load_and_save_data import load_pickle, save_pickle
from pathlib import Path

""" Functions used to run calculate_vectorfields code"""


def calculate_vector_fields(spike_rates_by_position_and_direction):

    bin_centres = spike_rates_by_position_and_direction['direction_bins'][:-1] + np.diff(spike_rates_by_position_and_direction['direction_bins'])/2

    vector_fields = {'units': {}, 'x_bins': spike_rates_by_position_and_direction['x_bins'], 
                     'y_bins': spike_rates_by_position_and_direction['y_bins'], 
                     'direction_bins': spike_rates_by_position_and_direction['direction_bins']}
    mean_resultant_lengths = {'units': {}, 'x_bins': spike_rates_by_position_and_direction['x_bins'], 
                     'y_bins': spike_rates_by_position_and_direction['y_bins'], 
                     'direction_bins': spike_rates_by_position_and_direction['direction_bins']} 
    

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
    derivatives_base: Path,
    unit_ids: list,
    vector_fields,
    goal_coordinates: list[int],
    x_centres: list,
    y_centres: list,
    plot_dir: Path,
    goals_to_include: list,
    methods: list,
    output_folder: Path,
):
    """ Plots the vectorfields and the consinks for all three methods.
    Note that the vector fields are the same for the three methods, but the consinks are different.
    
    Outputs
    ------
    derivatives_base / 'analysis' / 'cell_characteristics' / 'spatial_features' / 'vector_fields'/ f"vector_fields_unit_{u}.png"
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    METHOD_STYLE = {
        1: {
            "marker": "o",   # circle
            "label": "Method 1",
        },
        2: {
            "marker": "s",   # square
            "label": "Method 2",
        },
        3: {
            "marker": "^",   # triangle
            "label": "Method 3",
        },
    }
    
    hcoord, vcoord = get_coords_127sinks(derivatives_base)
    for u in unit_ids:
        fig, axes = plt.subplots(
            1, len(goals_to_include),
            figsize=(6 * len(goals_to_include), 6),
            squeeze=False
        )
        axes = axes[0]
        fig.suptitle(f"Vector fields – unit {u}", fontsize=20)

        lines = []
        for i, g in enumerate(goals_to_include):
            ax = axes[i]

            # ---------- title ----------
            if len(goals_to_include) > 1:
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
                    consink_plat = row['platform_g' + str(g)]
                    if np.isnan(consink_plat):
                        print(f"skipping {u} method {m}")
                        continue
                    pos = [hcoord[np.int32(consink_plat)- 1], vcoord[np.int32(consink_plat) - 1]]
                    ci95 = row[f'ci_95_g{g}']
                    #pos = row[f'position_g{g}']
                    ang = row[f'mean_angle_g{g}']

                    if not np.isfinite(mrl) or pos is None:
                        continue

                    if ang > np.pi:
                        ang -= 2 * np.pi

                    is_sig = np.isfinite(ci95) and mrl > ci95
                    color = 'red' if is_sig else 'grey'

                    style = METHOD_STYLE[m]
                    sig_str = "*" if is_sig else ""
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

                    """
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
                    """
                    lines.append(
                        f"M{m}: MRL={mrl:.2f}, CI95={ci95:.2f}, θ={np.rad2deg(ang):.1f}°{sig_str}"
                    )
            if lines:
                fig.text(
                    0.5,
                    0.90 - 0.04 * i,
                    f"Goal {g}:   " + "\n".join(lines),
                    ha="center",
                    va="top",
                    fontsize=11
                )
            if i == 0:
                ax.legend(frameon=False, loc='lower left')

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        output_path = plot_dir/ f"vector_fields_unit_{u}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        