import numpy as np
import os
import pandas as pd
import spikeinterface.extractors as se
from pathlib import Path
from tqdm import tqdm
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from spatial_features.utils.restrict_spiketrain_specialbehav import get_spike_train, restrict_spiketrain_specialbehav


def export_unit_spiketimes(derivatives_base: Path, goals_to_include: list = [0,1,2],add_speed_filt: bool = False, speed_threshold: float = 2, frame_rate: int = 25, sample_rate: int = 30000):
    """ For each unit, saves its spiketimes in frames in a dictionary. 
    If add_speed_filt = True, it also saves a dictionary with spiketimes for times when speed was above speed_threhshold
    
    Inputs
    -----
    derivatives_base (Path): Path to derivatives folder
    add_speed_filt (bool: False): Whether to additionally create a dictionary
                                  containing only spike times occurring when
                                  speed ≥ speed_threshold
    speed_threshold (float: 2): Speed threshold in cm/s used when add_speed_filt=True
    frame_rate (int: 25): Frame rate of camera (Hz)
    sample_rate (int: 30000): Electrophysiology sampling rate (Hz)

    Outputs
    -------
    derivatives_base / "analysis" / "cell_characteristics" / "unit_features" /
        "spike_frames.pkl"
            Pickle file containing:
                {unit_id: array of spike frames}

    If add_speed_filt = True:
        derivatives_base / "analysis" / "cell_characteristics" / "unit_features" /
        "spike_frames_speedfilt_{speed_threshold}cms.pkl"
            Pickle file containing:
                {unit_id: array of spike frames where speed ≥ threshold
    """
    print(f"Export unit spiketimes. Speedfilt = {add_speed_filt}")
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    kilosort_output_path = derivatives_base/ 'ephys'/ "concat_run"/"sorting"/ "sorter_output" 
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids

    spike_dict = {}
    spike_dict_speed_filt = {}
    spike_dict_g0 = {}
    spike_dict_g1 = {}
    spike_dict_g2 = {}
    spike_dict_g0_speed_filt = {}
    spike_dict_g1_speed_filt = {}
    spike_dict_g2_speed_filt = {}
    pos_folder = derivatives_base / "analysis/spatial_behav_data/XY_and_HD"

    # If you have one concatenated session file:
    df = pd.read_csv(pos_folder / "XY_HD_alltrials.csv")
    speed = df["speed"].values

    for unit_id in tqdm(unit_ids):
        spike_times = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_frames = spike_times * frame_rate / sample_rate
        spike_train_secs = spike_times / sample_rate
        spike_frames = spike_frames.astype(np.int64)
        spike_frames = spike_frames[spike_frames < len(speed)]
        spike_dict[unit_id] = spike_frames

        if add_speed_filt:
            # Get speed at each spike frame
            spike_speeds = speed[spike_frames]

            # Keep spikes where speed ≥ threshold
            valid_mask = spike_speeds >= speed_threshold

            spike_frames_filt = spike_frames[valid_mask]

            spike_dict_speed_filt[unit_id] = spike_frames_filt
            spike_train_filt_secs = spike_frames_filt  / frame_rate
            if len(spike_train_filt_secs) > len(spike_frames):
                breakpoint()
            
        for g in goals_to_include:
            spike_train_restricted = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder=rawsession_folder, goal=g)
            spike_train_restricted = spike_train_restricted * frame_rate
            spike_train_restricted= [np.int32(np.round(el)) for el in spike_train_restricted]
            if len(spike_train_restricted) > len(spike_frames):
                breakpoint()
            if g == 0:
                spike_dict_g0[unit_id] = spike_train_restricted
            elif g == 1:
                spike_dict_g1[unit_id] = spike_train_restricted
            else:
                spike_dict_g2[unit_id] = spike_train_restricted

            if add_speed_filt:
                spike_train_restricted = restrict_spiketrain_specialbehav(spike_train_filt_secs, rawsession_folder=rawsession_folder, goal=g)
                spike_train_restricted = spike_train_restricted * frame_rate
                spike_train_restricted= [np.int32(np.round(el)) for el in spike_train_restricted]
                if g == 0:
                    spike_dict_g0_speed_filt[unit_id] = spike_train_restricted
                elif g == 1:
                    spike_dict_g1_speed_filt[unit_id] = spike_train_restricted
                else:
                    spike_dict_g2_speed_filt[unit_id] = spike_train_restricted
   
    pos_folder = derivatives_base / "analysis/cell_characteristics/unit_features/spike_times"
    print(f"Saving to {pos_folder}")
    pos_folder.mkdir(exist_ok = True)
    np.save(pos_folder / 'spike_times_frames.npy', spike_dict)
    np.save(pos_folder / 'spike_times_speedfilt.npy', spike_dict_speed_filt) 

    if spike_dict_g0:
        np.save(pos_folder / 'spike_times_frames_g0.npy', spike_dict_g0)
        if spike_dict_g0_speed_filt:
            np.save(pos_folder / 'spike_times_frames_speedfilt_g0.npy', spike_dict_g0_speed_filt)

    if spike_dict_g1:
        np.save(pos_folder / 'spike_times_frames_g1.npy', spike_dict_g1)
        if spike_dict_g1_speed_filt:
            np.save(pos_folder / 'spike_times_frames_speedfilt_g1.npy', spike_dict_g1_speed_filt)

    if spike_dict_g2:
        np.save(pos_folder / 'spike_times_frames_g2.npy', spike_dict_g2)
        if spike_dict_g2_speed_filt:
            np.save(pos_folder / 'spike_times_frames_speedfilt_g2.npy', spike_dict_g2_speed_filt)
        
    breakpoint()
    

if __name__ == "__main__":
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-02_date-12022026\second_run_1602"
    derivatives_base = Path(derivatives_base)
    export_unit_spiketimes(derivatives_base, add_speed_filt = True, speed_threshold=2, goals_to_include = [1])
            
        