import numpy as np
import os
import glob
import pandas as pd


def concat_trials(derivatives_base):
    """
    Concatenates all trial files in the form {ratID}_{date}_g{trial}.csv into one dataframe and saves it as csv

    Args:
        rawsession_folder: path to the raw session folder
        
    Output:
        rawsession_folder/behaviour/concatenated_trials.csv: all trials concatenated
    """
    rawsession_folder = rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # List of all trial files
    trial_files = glob.glob(os.path.join(rawsession_folder, "behaviour", "*_g*.csv"))

    # Add each trial csv to df_list
    df_list = []
    for file in trial_files:
        df = pd.read_csv(file)
        trial_number = os.path.basename(file).split('_g')[-1].split('.csv')[0]  # Extract trial number from filename
        df['trial_number'] = int(trial_number) # Add trial number as a new column
        df_list.append(df)

    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Saving
    output_file = os.path.join(rawsession_folder, "behaviour", "concatenated_trials.csv")
    concatenated_df.to_csv(output_file, index=False)

    print(f"Concatenated trials saved to {output_file}")
    
    
if __name__ == "__main__":
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-02_date-05092025"
    concat_trials(rawsession_folder)