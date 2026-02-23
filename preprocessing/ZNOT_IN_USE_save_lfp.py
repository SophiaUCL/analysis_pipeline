import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import os 
import numpy as np
import json
from pathlib import Path
def save_lfp(derivatives_base: Path, sample_rate: int = 30000):
    """
    Looks at recording and saves lfp data to derivatives_base/ephys/lfp

    AI GENERATED, NOT TESTED, AND CURRENTLY (06/02/26) NOT IN USE
    Args:
        derivatives_base (Path): path to derivatives folder
        sample_rate (int: defaults to 30,000): sample rate of recording
    """
    spikeinterface_recording_path = os.path.join(derivatives_base, "ephys","concat_run","sorting", "spikeinterface_recording.json" )
    recording_path = os.path.join(derivatives_base, "ephys",  "concat_run", "preprocessed", "traces_cached_seg0.raw")
    output_path = os.path.join(derivatives_base, "ephys", "lfp")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    gain_to_uV, offset_to_uV = get_recording_data(spikeinterface_recording_path)
    rec = se.read_binary(
        file_paths = recording_path,
        # Info below is found in the json file in the same folder
        sampling_frequency=sample_rate,
        dtype = np.int16,  
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
        num_channels = 384,
    )
    # Apply a low-pass filter (e.g., below 300 Hz)
    lfp = sp.bandpass_filter(rec, freq_min=0.1, freq_max=300)
    # Optionally downsample to save space
    lfp = sp.resample(lfp, resample_rate=1000)  # 1 kHz typical LFP rate
    # Save to file
    lfp.save(folder=output_path, format="binary", overwrite = True)

def get_recording_data(spikeinterface_recording_path):
    """ 
    Gets gain_to_uV and offset_to_uV from the data file (which contains recording data)
    
    The path to the data for some reason keeps changing in different versions of spikeinterface
    If this format doesn't work, feed the file into chatGPT and then ask it to find the path. 
    Note that gain_to_uV and offset_to_uV are lists, so we take the first element 
    """
    # Obtain gain_to_uV and offset value
    with open(spikeinterface_recording_path, "r") as f:
        data = json.load(f)
        
    try:
        # Format option 1
        gain_to_uV = data["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording_list"][0]["properties"]["gain_to_uV"][0]
        offset_to_uV = data["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording_list"][0]["properties"]["offset_to_uV"][0]
    except (KeyError, IndexError, TypeError):
        try: # format option 2
            parent = data['kwargs']['parent_recording']
            properties = parent['properties']
            gain_to_uV = properties['gain_to_uV'][0]
            offset_to_uV = properties['offset_to_uV'][0]
        except (KeyError, IndexError, TypeError):
            # Format option 3
            gain_to_uV = data["properties"]["gain_to_uV"][0]
            offset_to_uV = data["properties"]["offset_to_uV"][0]
    
    print("gain_to_uV:", gain_to_uV)
    print("offset_to_uV:", offset_to_uV)
    
    return gain_to_uV, offset_to_uV

if __name__ == "__main__":
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    save_lfp(derivatives_base)