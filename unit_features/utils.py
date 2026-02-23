import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal
from pathlib import Path
from spikeinterface.core import BaseRecording, BaseSorting, SortingAnalyzer
from matplotlib.axes import Axes
from typing import Any

UnitTypes = Literal['pyramidal', 'good', 'all']
Tasks = Literal['hct', 'spatiotemporal']


def load_unit_ids(derivatives_base: Path, unit_type: UnitTypes, unit_ids: list) -> list:
    """ Returns unit_ids, the unit_ids that we will create rmaps for"""
    if unit_type == 'good':
        good_units_path = derivatives_base/ "ephys"/ "concat_run"/ "sorting"/"sorter_output"/ "cluster_group.tsv"
        good_units_df = pd.read_csv(good_units_path, sep='\t')
        unit_ids = good_units_df[good_units_df['group'] == 'good']['cluster_id'].values
        print("Using all good units")
    elif unit_type == 'pyramidal':
        pyramidal_units_path = derivatives_base/ "analysis"/ "cell_characteristics"/ "unit_features"/"all_units_overview"/ "pyramidal_units_2D.csv"
        pyramidal_units_df = pd.read_csv(pyramidal_units_path)
        pyramidal_units = pyramidal_units_df['unit_ids'].values
        unit_ids = pyramidal_units
        print("Using pyramidal units")
    elif unit_type == "all":
        print("Using all units")
        unit_ids = unit_ids
    else:
        raise ValueError("unit_type not good, pyramidal, or all. Provide correct input")
    return unit_ids


def get_trials_length_df(rawsession_folder: Path) -> pd.DataFrame:
    """ Returns trial_length_df"""
    # Get the data with trials length
    path_to_df = rawsession_folder/ 'task_metadata'/ 'trials_length.csv'
    if not path_to_df.exists():
        raise FileExistsError('trials_length.csv doesnt exist')
    trial_length_df = pd.read_csv(path_to_df)
    return trial_length_df

def get_goal_1_end_times(rawsession_folder: Path, trials_to_include: list, last_trial_openfield: bool = False) -> np.ndarray:
    """ Returns end times of goal 1 based on alltrials trial_day"""
    print("HCT: adding goal times to spikecount over trials")
    trialday_path = rawsession_folder/ 'behaviour'/ 'alltrials_trialday.csv'
    trialday_df  = pd.read_csv(trialday_path)
    if len(trialday_df) != len(trials_to_include) - last_trial_openfield:
        raise ValueError("length alltrials_trialday.csv is not the same as length trials to include. Remove unneeded trials")
    else:
        goal1_endtimes = np.array(trialday_df['Goal 1 end'])
    return goal1_endtimes

def get_spike_train_s(sorting: BaseSorting, unit_id: int, sample_rate: int = 30000) -> np.ndarray:
    """ Returns spiketrain in seconds for unit_id"""
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train = np.round(spike_train_unscaled/sample_rate) # spike times in seconds
    return spike_train

def get_total_trial_length(trials_to_include: list, trial_length_df: pd.DataFrame) -> float:
    """ Returns sum of all trials length in seconds"""
    total_trial_length = 0
    for tr in trials_to_include:
        trial_row = trial_length_df[(trial_length_df.trialnumber == tr)]
        trial_length = trial_row.iloc[0, 2]
        total_trial_length += trial_length   
    return total_trial_length

def make_plot(spike_train: np.ndarray, trial_starts: np.ndarray, trial_ends: np.ndarray, output_path: Path, n_bins: int, unit_id: int, goal1_endtimes = None, last_trial_openfield = False):
    """ Makes plot for unit_id and saves it"""
    fig =  plt.figure(figsize=(15, 5))
    plt.hist(spike_train, bins = np.int32(n_bins))

    # Vertical lines at trial boundaries
    for start in trial_starts[1:]:
        plt.axvline(x=start, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=trial_ends[-1], color='black', linestyle='--', linewidth=1)

    # Get current y-axis limits
    ymin, ymax = plt.ylim()

    # Label position: slightly below the top of the y-axis
    label_y = ymax

    # Place trial labels
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        mid = (start + end) / 2
        plt.text(mid, label_y, f'Trial {i+1}',
                ha='center', va='top', fontsize=9, color='black')
        if last_trial_openfield and i == len(goal1_endtimes):
            pass
        elif goal1_endtimes is not None:
            plt.axvspan(
                        start,
                        start + goal1_endtimes[i],
                        facecolor='lightblue',  # or 'lightblue'
                        alpha=0.5,
                        zorder = 0
                    )
    # Optional: adjust y-limit if you want more headroom
    plt.ylim(ymin, ymax * 1.05)
    plt.xlim(0, np.max(trial_ends))
    # Axis labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of spikes per minute")
    plt.title(f"Unit {unit_id}: Spike count across trials")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def get_unit_info(df_unit_metrics: pd.DataFrame, unit_id: int) -> tuple[float, str]:
    """ Loads unit firing rate and label for unit = unit_id"""
    row = df_unit_metrics[df_unit_metrics['unit_ids'] == unit_id]
    unit_firing_rate = row['firing_rate'].values[0]
    unit_label = row['label'].values[0]
    return unit_firing_rate, unit_label


def load_trial_xpos(pos_data_dir: Path, tr: int) -> np.ndarray:
    """ Returns x pos for trial tr"""
    trial_csv_path = pos_data_dir/  f'XY_HD_t{tr}.csv'
    xy_hd_trial = pd.read_csv(trial_csv_path)
                
    x = xy_hd_trial.iloc[:, 0].to_numpy()  
    return x 



def load_directories(derivatives_base: Path, rawsession_folder: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """ Loads paths and dfs that we need"""
        # Get directory for the positional data
    pos_data_dir = derivatives_base / "analysis/spatial_behav_data/XY_and_HD"
    if not pos_data_dir.exists():
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")

    epoch_times = pd.read_csv(rawsession_folder / "task_metadata/epoch_times.csv")

    df_unit_metrics = pd.read_csv(
        derivatives_base / "analysis/cell_characteristics/unit_features/all_units_overview/unit_metrics.csv"
    )

    trials_length = pd.read_csv(rawsession_folder / "task_metadata/trials_length.csv")

    output_dir = derivatives_base / "analysis/cell_characteristics/unit_features/epochs_spike_count"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return pos_data_dir, epoch_times, df_unit_metrics, trials_length, output_dir



def get_trial_length_info(epoch_times: pd.DataFrame, trials_length: pd.DataFrame,  tr: int) -> tuple[float, float, pd.DataFrame]:
    """ Returns start time of trial and trial length"""
    trial_row = epoch_times[(epoch_times.trialnumber == tr)]
    start_time = trial_row.iloc[0, 1]

    trial_length_row = trials_length[(trials_length.trialnumber == tr)]
    trial_length = trial_length_row.iloc[0, 2]
    return start_time, trial_length, trial_row
            
def get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate):
    """ Restricts spiketrain to current trial"""
    spike_train_this_trial = np.copy(spike_train)
    spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)] # filtering for current trial
    spike_train_this_trial = [el - trial_dur_so_far for el in spike_train_this_trial]
    spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)/frame_rate]
    return spike_train_this_trial


def plot_trial_firing(spike_train_this_trial: np.ndarray,trial_row:pd.DataFrame, n_epochs: int, tr: int, x: np.ndarray, frame_rate: int, ax: Axes):
    """ Makes subplot of firing for one trial per each epoch"""
    ax.set_title(f"Trial {tr} | n = {len(spike_train_this_trial)} spikes")
     # Plot histogram of spike times
    ax.hist(
        spike_train_this_trial,
        bins=50,
        range=(0, len(x)/frame_rate),
        color='black',
        alpha=0.7,
        zorder = 2
    )

    # Plot dotted lines for epoch start and end
    for e in range(1, n_epochs + 1):
        epoch_start = trial_row.iloc[0, 2*e - 1]
        epoch_end = trial_row.iloc[0, 2*e]

        ax.axvspan(
            epoch_start,
            epoch_end,
            facecolor='lightblue',  # or 'lightblue'
            alpha=0.5,
            zorder = 0
        )
        epoch_mid = (epoch_start + epoch_end)/ 2

        # Add text label "Epoch {e}" at the midpoint
        ax.text(
            epoch_mid,
            ax.get_ylim()[1] *0.95,  # slightly above the top of the histogram
            f"Epoch {e}",
            ha='center',
            va='bottom',
            fontsize=8,
        )
    

    ax.set_xlim(0, len(x)/frame_rate)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike count")       
    
    