import numpy as np
import os
import glob
import pandas as pd
import spikeinterface.extractors as se
from tqdm import tqdm
from astropy.stats import circmean
from spatiotemporal_analysis.get_sig_cells import get_sig_cells # CHECK IF WORKS
from spatiotemporal_analysis.utils import resultant_vector_length
from pathlib import Path

def get_MRL_significance(derivatives_base: Path, trials_to_include, frame_rate = 25, sample_rate = 30000, num_bins = 24):
    """
    Gets the significance level of the MRL value for each trial epoch for each unit
    Uses same method as spatiotemporal code

    Inputs:
    derivatives_base: path to derivatives folder
    rawsession_folder: path to rawsession folder
    trials_to_include: trials for this recording day
    frame_rate: frame_rate of camera (default = 25)
    sample_rate: sample rate of recording device (default = 30000)
    num_bins: number of bins used to bin the spatial data (default = 24, giving 15 degree bins)
    
    Returns:
    Path to df with MRL data for all units (which can be used in roseplot)
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # For plotting
    n_epochs = 3

    # In this df the directional data of all units will be saved
    directional_data_all_units = pd.DataFrame(
        columns=[
            'cell', 'trial', 'epoch', 'MRL', 'mean_direction',
            'percentiles95', 'percentiles99', 'significant', 'num_spikes'
        ]
    )

    # Load data files
    kilosort_output_path = os.path.join(derivatives_base,  "concat_run","sorting", "sorter_output" )
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    unit_ids = sorting.unit_ids
    labels = sorting.get_property('KSLabel')
    good_units_ids = [el for el in unit_ids if labels[el] == 'good']

    # Get directory for the positional data
    pos_data_dir = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD')
    if not os.path.exists(pos_data_dir):
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")
    
    csv_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.csv'))
    if len(csv_path) > 0:
        epoch_times_allcols = pd.read_csv(csv_path[0], header=None)
    else:
        excel_path = glob.glob(os.path.join(rawsession_folder, 'task_metadata', 'behaviour*.xlsx'))
        if len(excel_path) > 0:
            epoch_times_allcols = pd.read_excel(excel_path[0], header=None)
        else:
            raise FileNotFoundError('No behaviour CSV or Excel file found in the specified folder.')


    # loading dataframe with unit information
    path_to_df = os.path.join(derivatives_base, "analysis", "cell_characteristics", "unit_features", "all_units_overview","unit_metrics.csv")
    df_unit_metrics = pd.read_csv(path_to_df) 


    epoch_times= epoch_times_allcols.iloc[:, [10, 12, 14, 16, 18]]
    epoch_times.columns = ['epoch 1 end', 'epoch 2 start', 'epoch 2 end', 'epoch 3 start', 'epoch 3 end']
    epoch_times.insert(0, "epoch 1 start", np.zeros(len(epoch_times)))
    epoch_times.insert(0,'trialnumber',  trials_to_include)

    trials_length_path = os.path.join(rawsession_folder, 'task_metadata', 'trials_length.csv')
    trials_length = pd.read_csv(trials_length_path)
        
    # Output folder
    output_folder_data = os.path.join(derivatives_base, 'analysis', 'cell_characteristics', 'spatial_features', 'spatial_data')
        
    if not os.path.exists(output_folder_data):
        os.makedirs(output_folder_data)
    

    # Looping over all units
    for unit_id in tqdm(unit_ids):

        # Loading data
        spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train = np.round(spike_train_unscaled*frame_rate/sample_rate) # trial data is now in frames in order to match it with xy data

        # Duration of trial (starts at 0)
        trial_dur_so_far = 0 # NOTE: There may be errors if trial 1 (or g0) is excluded from analysis

        # Looping over trials
        for tr in trials_to_include:
            spike_train_this_trial = np.copy(spike_train)

            # Trial times
            trial_row = epoch_times[(epoch_times.trialnumber == tr)]
            start_time = trial_row.iloc[0, 1]
            trial_length_row = trials_length[(trials_length.trialnumber == tr)]
            trial_length = trial_length_row.iloc[0, 2]


            # Positional data
            trial_csv_name = f'XY_HD_t{tr}.csv'
            trial_csv_path = os.path.join(pos_data_dir, trial_csv_name)
            xy_hd_trial = pd.read_csv(trial_csv_path)            
            x = xy_hd_trial.iloc[:, 0].to_numpy()
            y = xy_hd_trial.iloc[:, 1].to_numpy()
            hd = xy_hd_trial.iloc[:, 2].to_numpy()
            hd_rad = np.deg2rad(hd)

            # Length of trial
            spike_train_this_trial =  [el for el in spike_train_this_trial if el > np.round(trial_dur_so_far+ start_time)*frame_rate] # filtering for current trial
            spike_train_this_trial = [el - trial_dur_so_far*frame_rate for el in spike_train_this_trial] # setting 0 as start of trial
            spike_train_this_trial = [el for el in spike_train_this_trial if el < len(x)] # We're not plotting more than the spatial data we have

            # Make plots
            # Obtaining ratemap data

            trial_dur_so_far += trial_length

            # Looping over epochs
            for e in range(1, n_epochs + 1):
                # Obtain epoch start and end times
                epoch_start = trial_row.iloc[0, 2*e-1]
                epoch_end = trial_row.iloc[0, 2*e]

                spike_train_this_epoch = [np.int32(el) for el in spike_train_this_trial if el > frame_rate*epoch_start and el < frame_rate *epoch_end]
                spike_train_this_epoch = np.asarray(spike_train_this_epoch, dtype=int)

                
                # HD calculations
                if len(spike_train_this_epoch) > 0:
                    # Obtaining hd for this epoch and calculating how much the animal sampled in each bin
                    hd_this_epoch = hd[np.int32(epoch_start*frame_rate):np.int32(epoch_end*frame_rate)]
                    hd_this_epoch = hd_this_epoch[~np.isnan(hd_this_epoch)]
                    hd_this_epoch_rad = np.deg2rad(hd_this_epoch)
                    occupancy_counts, _ = np.histogram(hd_this_epoch_rad, bins=num_bins, range = [-np.pi, np.pi])
                    occupancy_time = occupancy_counts / frame_rate 

                    # Getting the spike times and making a histogram of them
                    spikes_hd = hd[spike_train_this_epoch]
                    spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
                    spikes_hd_rad = np.deg2rad(spikes_hd)
                    counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

                    # Calculating directional firing rate
                    direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)

                    # Getting significance
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    MRL = resultant_vector_length(bin_centers, w = direction_firing_rate)
                    percentiles_95_value, percentiles_99_value, MRL_values = get_sig_cells(spike_train_this_epoch, hd_rad, epoch_start*frame_rate, epoch_end*frame_rate, occupancy_time, n_bins = num_bins)
                    breakpoint()
                    mu = circmean(bin_centers, weights = direction_firing_rate)
                    # Add significance data for every element (even if not significant)
                    new_element = {
                        'cell': unit_id,
                        'trial': tr,
                        'epoch': e,
                        'MRL': MRL,
                        'mean_direction': np.rad2deg(mu),
                        'percentiles95': percentiles_95_value,
                        'percentiles99': percentiles_99_value,
                        'significant': 'ns', 
                        'num_spikes': len(spike_train_this_epoch)
                    }

                    if MRL > percentiles_95_value:
                        new_element['significant'] = 'sig'
                    directional_data_all_units.loc[len(directional_data_all_units)] = new_element

    # Saving directional data
    output_path = os.path.join(output_folder_data, f"directional_tuning_{np.int32(360/num_bins)}_degrees_max58shift.csv")
    directional_data_all_units.to_csv(output_path)
    print(f"Data saved to {output_path}")
    return output_path, np.int32(360/num_bins)

if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-01_date-30072025\all_trials"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-01_date-30072025"
    trials_to_include = np.arange(1,11)

    get_MRL_significance(derivatives_base, rawsession_folder, trials_to_include, frame_rate = 25, sample_rate = 30000)
    