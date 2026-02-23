Folder: unit_features


Contains functions that create plot or data that goes in the cell_characteristics/unit_features folder
These are called from spatial_processing_pipeline


plot_spikecount_over_trials: plots the spikecount of each unit for all the trial, separating each trial using line. For HCT, also denotes start and end of the task
Outputs:
	derivatives_base\analysis\cell_characteristics\unit_features\spikecount_over_trials\unit_{}_sc_over_trials.png: shows the spike counts over trials

plot_firing_each_epoch: used for spatiotemporal task. For each unit, creates an n by 3 plot showing the firing rate for each trial with each epoch indicated. 
	derivatives_base /analysis/cell_characteristics/unit_features/epochs_spike_count/unit_{unit_id}_epoch_firing.png
        firing of each unit for each epoch in each trial

test_restrict_spiketrain:File that overlays that spike times of a unit with the goal intervals, to visually check whether the spiketrain restriction by goal worked as expected.
utils: contains function used by above two functions to run the code
Saves: derivatives_base/'analysis'/ 'cell_characteristics'/'unit_features'/ 'individual_spiketimes_goals'


get_spiketimes_allunits: Function used to create npy files for each trial, containing an array for each unit for spiketimes for that trial in frames
derivatives_base / "analysis" / "cell_characteristics" / "unit_features" / "spikes_alltrials"/ f"spikes_tr{tr}_{frame_rate}fps.npy"
        npy array with an array for each unit containing its spike times for that trial in frames

classify_cells: Contains two functions to classify cells, one based just on peak to valley time and one based on peak to valley time and mean firing rate. Both use k-means clustering. CURRENTLY NOT IN USE (06/02/26), Chiara is creating  more sophisticated way.
Outputs unit_features/all_units_overview/pyramidal_units_2D.csv, which is the main way to find out which units are pyramidal!!!! (used in all load_units functions)