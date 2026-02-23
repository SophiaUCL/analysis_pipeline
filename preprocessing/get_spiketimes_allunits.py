import numpy as np
from pathlib import Path
import pandas as pd
import spikeinterface.extractors as se
from spatial_features.utils.spatial_features_utils import get_spikes_tr, get_spike_train_frames
import json
from tqdm import tqdm

def get_spiketimes_alltrials(derivatives_base: Path, frame_rate: int = 25) -> None:
    """ Function used to create npy files for each trial, containing an array for each unit for spiketimes for that trial in frames
    Inputs
    -----
    derivatives_base (Path): Path to derivatives folder
    frame_rate (int: 25): frame rate of camera
    
    Outputs:
    -------
    derivatives_base / "analysis" / "cell_characteristics" / "unit_features" / "spikes_alltrials"/ f"spikes_tr{tr}_{frame_rate}fps.npy"
        npy array with an array for each unit containing its spike times for that trial in frames
        
    Used for:
    --------
    Overlaying spiketimes on video (Eylon's GUI)
    """
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
    
    print(f"Saving spikes to {output_folder}")
    for tr in trial_numbers:
        print(f"Currently at trial {tr} out of {len(trial_numbers)}")
        pos_data_path = derivatives_base /"analysis"/ "spatial_behav_data" / "XY_and_HD" / f"XY_HD_t{tr}.csv"
        pos_data = pd.read_csv(pos_data_path)
        
        trial_dur_so_far = cumul_length_alltrials[tr-1]
        
        spikes_this_trial =  np.ndarray(dtype=object, shape=(len(unit_ids),))
        for i, unit_id in tqdm(enumerate(unit_ids)):
            # Load spiketrain
            spike_train_fr = get_spike_train_frames(sorting, unit_id, sample_rate = 30000, frame_rate = frame_rate)
            
            # Restrict to this trial
            spike_train_tr = get_spikes_tr(spike_train_fr, trial_dur_so_far, start_time = 0, x = pos_data, frame_rate = 25)
            
            spikes_this_trial[i] = spike_train_tr
        output_file = output_folder / f"spikes_tr{tr}_{frame_rate}fps.npy"
        np.save(output_file, spikes_this_trial)
    
if __name__ == "__main__":
    derivatives_base = Path(r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801")
    get_spiketimes_alltrials(derivatives_base)