import numpy as np
from pathlib import Path
import pandas as pd
import spikeinterface.extractors as se
import json
from tqdm import tqdm

def get_spiketimes_alltrials(derivatives_base: Path, speed_filt: bool = False, frame_rate: int = 25) -> None:
    """ Function used to create npy files for each trial, containing an array for each unit for spiketimes for that trial in frames
    Inputs
    -----
    derivatives_base (Path): Path to derivatives folder
    speed_filt (bool: False): Whether we want to use the speed filtered spiketimes
    frame_rate (int: 25): frame rate of camera
    
    Outputs:
    -------
    derivatives_base / "analysis" / "cell_characteristics" / "unit_features" / "spikes_alltrials"/ f"spikes_tr{tr}_{frame_rate}fps.npy"
        npy array with an array for each unit containing its spike times for that trial in frames
        
    Used for:
    --------
    Overlaying spiketimes on video (Eylon's GUI)
    """
    if speed_filt:
        print("Filtering by speed")
    else:
        print("Not filtering by speed")
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Load trials
    config_path = derivatives_base / "config.json"
    with open(config_path) as json_data:
        configs = json.load(json_data)
        json_data.close()
    inputs = configs["inputs"]
    trial_numbers = inputs["trial_numbers"]
    
    
    # Load units
    kilosort_output_path = derivatives_base /  'ephys' / "concat_run"/ "sorting"/ "sorter_output"
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    
    # Load trial lengths
    trial_length_path = rawsession_folder / "task_metadata" / "trials_length.csv"
    trials_length = pd.read_csv(trial_length_path)
    cumul_length_alltrials = trials_length["cumulative length"].to_numpy()
    
    # output path
    output_folder = derivatives_base / "analysis" / "cell_characteristics" / "unit_features" / "spikes_alltrials"
    output_folder.mkdir(exist_ok = True)
    
    spike_dict = get_spiketrain_from_dict(derivatives_base, speed_filt = speed_filt)

    print(f"Saving spikes to {output_folder}")
    for tr in trial_numbers:
        print(f"Currently at trial {tr} out of {len(trial_numbers)}")
        pos_data_path = derivatives_base /"analysis"/ "spatial_behav_data" / "XY_and_HD" / f"XY_HD_t{tr}.csv"
        pos_data = pd.read_csv(pos_data_path)
        
        trial_dur_so_far = cumul_length_alltrials[tr-1]
        
        spikes_this_trial =  {}
        for unit_id in tqdm(unit_ids):
            # Load spiketrain
            spike_train_fr = spike_dict[unit_id]
            
            # Restrict to this trial
            spike_train_tr = get_spikes_tr(spike_train_fr, trial_dur_so_far, start_time = 0, x = pos_data, frame_rate = 25)
            
            spikes_this_trial[unit_id] = spike_train_tr
        output_file = output_folder / f"spikes_tr{tr}_{frame_rate}fps.npy" if not speed_filt else output_folder / f"spikes_tr{tr}_{frame_rate}fps_speedfilt.npy"
        np.save(output_file, spikes_this_trial)

def get_spiketrain_from_dict(derivatives_base: Path, speed_filt: bool = False):
    """ Gets spiketrain dictionary"""
    pos_folder = derivatives_base / "analysis/cell_characteristics/unit_features/spike_times"
    print(f"Saving to {pos_folder}")
    pos_folder.mkdir(exist_ok = True)
    
    if speed_filt:
        spike_dict = np.load(pos_folder / 'spike_times_speedfilt.npy', allow_pickle=True).item()
    else:
        spike_dict = np.load(pos_folder / 'spike_times_frames.npy', allow_pickle = True).item()
    return spike_dict

def get_spikes_tr(spike_train, trial_dur_so_far, start_time, x, frame_rate = 25):
    """ Restricts spiketrain to current trial
    Expects input in frames"""
    spike_train_this_trial = np.copy(spike_train)
    spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)*frame_rate] # filtering for current trial
    spike_train_this_trial = [el - np.round(trial_dur_so_far*frame_rate) for el in spike_train_this_trial]
    spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)]
    return spike_train_this_trial
    
if __name__ == "__main__":
    derivatives_base = Path(r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-02_date-12022026\second_run_1602")
    speed_filt = False
    get_spiketimes_alltrials(derivatives_base, speed_filt = speed_filt)