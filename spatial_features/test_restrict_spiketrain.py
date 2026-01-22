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
from utils.spatial_features_utils import  get_ratemaps_restrictedx, load_unit_ids, get_outline, get_limits, get_posdata, get_occupancy_time, get_ratemaps, get_spike_train_frames, get_directional_firingrate
from utils.spatial_features_plots import plot_rmap, plot_occupancy, plot_directional_firingrate
from utils.restrict_spiketrain_specialbehav import get_spike_train, restrict_spiketrain_specialbehav

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
from tqdm import tqdm

def load_goal_intervals_frames(rawsession_folder, frame_rate):
    """
    Returns goal 1 and goal 2 intervals in FRAMES.
    """
    path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")
    df = pd.read_csv(path)

    # goal 1: cols 2,3
    g1_intervals_sec = list(zip(df.iloc[:, 2], df.iloc[:, 3]))
    g2_intervals_sec = list(zip(df.iloc[:, 4], df.iloc[:, 5]))

    # convert to frames
    g1_intervals_fr = [
        (int(np.floor(s * frame_rate)), int(np.ceil(e * frame_rate)))
        for s, e in g1_intervals_sec
    ]
    g2_intervals_fr = [
        (int(np.floor(s * frame_rate)), int(np.ceil(e * frame_rate)))
        for s, e in g2_intervals_sec
    ]

    return g1_intervals_fr, g2_intervals_fr


def plot_spiketimes_with_goal_shading(
    sorting,
    unit_ids,
    derivatives_base,
    output_folder,
    show_plots = True,
    sample_rate = 30000,
    frame_rate = 25,
    include_g0 = True
):
    """
    Plot spike times in FRAMES with shaded goal intervals.

    Spikes:
      black → g3 (full)
      blue  → g1
      red   → g2
      green → g0 (optional)

    Shading:
      light blue → goal 1 intervals
      light red  → goal 2 intervals
    """  
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    # full trial position only for length
    x_full, _, _, _ = get_posdata(derivatives_base, method="ears", g=3)
    n_frames = len(x_full)

    # load goal intervals (frames)
    g1_intervals, g2_intervals = load_goal_intervals_frames(
        rawsession_folder, frame_rate
    )

    print("Plotting spike times with goal shading (frames)")

    for unit_id in tqdm(unit_ids):
        
        spikes_full = get_spike_train(
            sorting, sample_rate, rawsession_folder,
            unit_id, g=3, frame_rate=frame_rate, pos_data=x_full
        )

        spikes_g1 = get_spike_train(
            sorting, sample_rate, rawsession_folder,
            unit_id, g=1, frame_rate=frame_rate, pos_data=x_full
        )

        spikes_g2 = get_spike_train(
            sorting, sample_rate, rawsession_folder,
            unit_id, g=2, frame_rate=frame_rate, pos_data=x_full
        )

        if include_g0:
            spikes_g0 = get_spike_train(
                sorting, sample_rate, rawsession_folder,
                unit_id, g=0, frame_rate=frame_rate, pos_data=x_full
            )

        fig, ax = plt.subplots(figsize=(14, 3))

        # ---- shaded goal intervals ----
        for s, e in g1_intervals:
            ax.axvspan(s, e, color="lightblue", alpha=0.4, zorder=0)

        for s, e in g2_intervals:
            ax.axvspan(s, e, color="mistyrose", alpha=0.4, zorder=0)

        # ---- spike rasters ----
        y = 0.0
        ax.vlines(spikes_full, y, y + 1.0, color="black", linewidth=0.8, label="g3 (full)")
        y += 1.2


        ax.vlines(spikes_g2, y, y + 1.0, color="red", linewidth=1.2, label="g2")
        y += 1.2
        
        ax.vlines(spikes_g1, y, y + 1.0, color="blue", linewidth=1.2, label="g1")
        y += 1.2

        if include_g0:
            ax.vlines(spikes_g0, y, y + 1.0, color="green", linewidth=1.2, label="g0")

        info_lines = [
            f"g3 (full): {len(spikes_full)}",
            f"g1: {len(spikes_g1)}",
            f"g2: {len(spikes_g2)}",
        ]

        if include_g0:
            info_lines.append(f"g0: {len(spikes_g0)}")

        info_text = "\n".join(info_lines)

        ax.text(
            0.995, 0.98,
            info_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

        ax.set_xlim(0, n_frames)
        ax.set_ylim(-0.2, y + 1.2)
        ax.set_xlabel("Frame")
        prop = (len(spikes_g1) + len(spikes_g2)) / len(spikes_full) * 100 if len(spikes_full) > 0 else 0
        ax.set_title(f"Unit {unit_id}, Spikes\nProportion of spikes in g1+g2: {prop:.2f}%")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False
        )

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"unit_{unit_id}_spiketimes_goals.png"))
        if show_plots:
            plt.show()

def test_restrict_spiketrain(derivatives_base, unit_type, include_g0 = True, show_plots = True):
    
    # Load data files
    kilosort_output_path = os.path.join(derivatives_base, 'ephys', "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    
    # Output folder
    output_folder = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'unit_features', 'individual_spiketimes_goals')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Here unit ids get filtered by unit_type
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)

    print(f"Saving figures to {output_folder}")
    plot_spiketimes_with_goal_shading(
        sorting,
        unit_ids,
        derivatives_base,
        show_plots = show_plots,
        output_folder = output_folder,
        sample_rate = 30000,
        frame_rate = 25,
        include_g0 = include_g0
    )

if __name__ == "__main__":
    # Example usage
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    unit_type = "all"
    test_restrict_spiketrain(derivatives_base, unit_type, include_g0=True, show_plots = False)
