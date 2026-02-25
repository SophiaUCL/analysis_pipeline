'''
This script is the main pipeline used for the data analysis
Prior to running, install the environment processing-pipeline (as can be found in environment_processing.yml)

'''
import numpy as np
import datetime
from preprocessing.spikewrap import run_spikewrap
from preprocessing.zero_pad_trials import zero_pad_trials
from preprocessing.get_length_all_trials import get_length_all_trials
from preprocessing.postprocessing_spikeinterface import run_spikeinterface
from preprocessing.append_config import append_config
from preprocessing.find_paths import find_paths
from preprocessing.make_epoch_times_csv import make_epoch_times_csv
from unit_features.export_unit_spiketimes import export_unit_spiketimes
from unit_features.get_spiketimes_allunits import get_spiketimes_alltrials
import torch
from pathlib import Path
# basic processing preprocesses the data and makes spatial plots

# Currently using processing_pipeline, not processing_pipelin2
user = "Sophia"
task = "hct"
base_path = r"E:\Honeycomb_task_1g"
base_path = Path(base_path)
subject_number = "001"
session_number = "02"
trial_session_name = "second_run_1602" 
trial_numbers = np.arange(1,14)
goals_to_include = [1]
if len(trial_numbers) == 1:
    concat_runs = False
else:
    concat_runs = True
frame_rate = 25
sample_rate = 30000
    
# Verify GPU is used
print(torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version torch was built with:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise ValueError("GPU not engaged, rerun with or update torch")
# === Finding the subject folder and session name ===
derivatives_base, rawsession_folder, rawsubject_folder, session_name = find_paths(base_path, subject_number, session_number, trial_session_name)


# === Adding data to config file ===
config_data = {
    'inputs': {
        'name': user,
        'date': str(datetime.date.today()),
        'time': str(datetime.datetime.now().time()),
        'task': task,
        'base_path': str(base_path),
        'subject_number': subject_number,
        'session_number': session_number,
        'trial_session_name': trial_session_name,
        'trial_numbers': trial_numbers.tolist()
    }
}
# === Adding data to config file ===
append_config(derivatives_base, config_data)

# === Zero padding trials ===
zero_pad_trials(rawsession_folder)

if task == "spatiotemp":
    make_epoch_times_csv(derivatives_base, trial_numbers)
# == Obtain length for all of the trials, making a csv out of its === 
get_length_all_trials(derivatives_base, trial_numbers)


# === Running Spikewrap preprocessing ===
run_spikewrap(derivatives_base, rawsubject_folder, session_name, concat_runs = concat_runs)

# === Post processing ===
run_spikeinterface(derivatives_base, run_analyzer_from_memory=False, run_df_from_memory=False, clear_plot_folder=False)

# === Getting spiketimes for all the cells exported =====
export_unit_spiketimes(derivatives_base, goals_to_include, add_speed_filt = False, frame_rate = frame_rate, sample_rate = sample_rate)


# ==== Getting the spiketimes per trial for each unit exported
get_spiketimes_alltrials(derivatives_base, speed_filt = False, frame_rate = frame_rate)