Folder: preprocessing

Contains functions used by main_pipeline.py in the initial stages of processing

Specific functions:
append_config: called by main_pipeline in order to save the session variables to derivatives_base/config.json. Also contains a function called deep_update in order to add extra variables to the config files

find_paths: called by main_pipeline. used to find the paths to the derivatives_base and rawsession_folder and to create the derivatives_base folder.
Returns:
        rawsession_folder (Path): path to the raw session folder
        derivatives_base (Path): path to the derivatives folder
        rawsubject_folder (Path): path to the raw subject folder
        session_name (str): name of the session


get_length_all_trials: called by main_pipeline.py. Creates a file with the trial length for all the trials (which it obtains from the meta files)
Outputs:
	rawsession_folder/task_metadata/trials_length.csv
    	csv file with trial numbers, g numbers, trial length (s), and cumulative length (length of all trials up until that trial)

spikewrap: called by main_pipeline.py. Does everything preprocessing related, runs spikewrap and kilosort. 
Creates:
	kilosort files
	binned data files

zero_pad_trials: called by main_pipeline.py. Ensures that in the ephys folder, the folder are named g01, g02, etc instead of g1, g2

postprocessing_spikeinterface: Creates df with unit metrics, figures with autocorrelograms and waveforms etc etc

make_epoch_times_csv: Creates a df with the epoch times 

spikeinterface_utils: contains all functins used for postprocessing spikeinterface
Functions that are not in use:

find_trial_numbers: NOT IN USE AS OF 05/02/26. Finds the trial numbers that we use in our trials

save_lfp: NOT IN USE (06/02) Saves lfp data. Never tested and AI generated
