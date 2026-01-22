from find_popsink_main import plot_popsink_w_vectors, calculate_popsink
from utilities.trials_utils import get_limits_from_json, get_goal_numbers, get_coords, get_sink_positions_platforms, get_coords_127sinks
import numpy as np
import os
import matplotlib.pyplot as plt
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.trials_utils import get_goal_coordinates, get_goal_numbers, get_coords_127sinks, get_unit_ids, get_pos_data, verify_allnans, get_spike_train, get_sink_positions_platforms, translate_positions
import matplotlib
matplotlib.use("QtAgg")
from plotting.plot_sinks import plot_all_consinks_onegoal_127sinks


def plot_consink_and_popsink(derivatives_base, unit_type, run_zero= True):
    """ Makes a plot with the consinks and the population sinks (2 methods)"""

    # plotting variables
    nrows = 4
    ncols = 2 + run_zero
    fig, axs = plt.subplots(nrows, ncols, figsize = (ncols * 6, nrows * 6))

    # output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features',
                                 'con_and_popsink')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    popsink_folder =os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features',
             'popsinks')
    consink_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features',
                                  'consinks')



    # Other variables
    methods = ['trial_norm', 'plat_norm']
    limits = get_limits_from_json(derivatives_base)
    goal_numbers = get_goal_numbers(derivatives_base)
    hcoord, vcoord = get_coords(derivatives_base)
    hcoord_127, vcoord_127 = get_coords_127sinks(derivatives_base)
    platforms_trans =translate_positions()
    x_diff = np.mean(np.diff(hcoord))
    y_diff = np.mean(np.diff(vcoord))
    jitter = (x_diff / 3, y_diff / 3)
    for g in range(3):
        if g == 0 and not run_zero:
            continue
        for s in range(2): # s == 0: consink. s == 1: popsink
            if s == 0:
                for method in [1,2]:
                    # Check if consinks_df is a dictionary otherwise convert
                    try:
                        consinks_df = load_pickle(f'consinks_df_m{method}', consink_folder)
                        average_sink = load_pickle(f'average_sink_m{method}', consink_folder)
                    except:
                        main(derivatives_base, 'all trials', unit_type = unit_type, method=method, code_to_run=[-1, 0, 1, 2],
                             include_g0=run_zero)
                        consinks_df = load_pickle(f'consinks_df_m{method}', consink_folder)
                        average_sink = load_pickle(f'average_sink_m{method}', consink_folder)
                    plot_all_consinks_onegoal_127sinks(consinks_df, g, method, goal_numbers, hcoord_127, vcoord_127,
                                                           platforms_trans, jitter, average_sink, ax=axs[method - 1, g - 1 + run_zero])

            if s == 1:
                for m, method in enumerate(methods):
                    if g == 3:
                        name = f"popsink_wholetrial_{method}"
                        title = f"G1 + G2, {method}"
                    else:
                        name = f"popsink_g{g}_{method}"
                        title = f"Goal {g}, {method}"

                    try:
                        mrl_dataset = load_pickle(name, popsink_folder)
                    except:
                        calculate_popsink(derivatives_base, unit_type=unit_type, code_to_run=[1],
                                          run_zero=False, frame_rate=25, sample_rate=30000)
                        mrl_dataset = load_pickle(name, output_folder)
                    plot_popsink_w_vectors(mrl_dataset, hcoord, vcoord, limits, output_folder, plot_name=title, ax=axs[ 2 + m, g - 1 + run_zero])
    plt.savefig(os.path.join(output_folder, "overview.png"))
    print(f"Saved plot to {os.path.join(output_folder, 'overview.png')}")
    plt.show()


if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    plot_consink_and_popsink(derivatives_base, "pyramidal", run_zero = False)
