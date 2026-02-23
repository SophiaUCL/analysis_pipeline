import pandas as pd
from pathlib import Path

def concat_trials(derivatives_base: Path):
    """
    Concatenates all trial files in the form {ratID}_{date}_g{trial}.csv into one dataframe and saves it as csv

    Args:
    derivatives_base (Path): path to the derivatives folder
        
    Output:
        rawsession_folder/behaviour/concatenated_trials.csv: all trials concatenated
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    behaviour_folder = rawsession_folder / "behaviour"
    # List of all trial files
    trial_files = list(behaviour_folder.glob("*_g*.csv"))

    # Add each trial csv to df_list
    df_list = []
    for file in trial_files:
        df = pd.read_csv(file)
        base_path = file.name
        trial_number = base_path.split('_g')[-1].split('.csv')[0]  # Extract trial number from filename
        df['trial_number'] = int(trial_number) # Add trial number as a new column
        df_list.append(df)

    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Saving
    output_file = behaviour_folder / "concatenated_trials.csv"
    concatenated_df.to_csv(output_file, index=False)

    print(f"Concatenated trials saved to {output_file}")
    
    
if __name__ == "__main__":
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
    derivatives_base = Path(derivatives_base)
    concat_trials(derivatives_base)