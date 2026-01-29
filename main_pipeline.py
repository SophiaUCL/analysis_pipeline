'''
This script is the main pipeline used for the data analysis
Prior to running, install the environment processing-pipeline (as can be found in environment_processing.yml)

'''
import numpy as np
import datetime
from preprocessing.spikewrap import run_spikewrap
from preprocessing.zero_pad_trials import zero_pad_trials
from preprocessing.get_length_all_trials import get_length_all_trials
from unit_features.postprocessing_spikeinterface import run_spikeinterface
from other.append_config import append_config
from other.find_paths import find_paths
# basic processing preprocesses the data and makes spatial plots

# Currently using processing_pipeline, not processing_pipelin2
user = "Sophia"
task = "hct"
base_path = r"E:\Honeycomb_task_1g"
subject_number = "001"
session_number = "01"
trial_session_name = "first_run_2801" 
trial_numbers = np.arange(1,27)
if len(trial_numbers) == 1:
    concat_runs = False
else:
    concat_runs = True
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

# == Obtain length for all of the trials, making a csv out of its === 
get_length_all_trials(rawsession_folder, trial_numbers)


# === Running Spikewrap preprocessing ===
run_spikewrap(derivatives_base, rawsubject_folder, session_name, concat_runs = concat_runs)

# === Post processing ===
run_spikeinterface(derivatives_base, True, True)

