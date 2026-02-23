import numpy as np
import pandas as pd
from pathlib import Path

def create_intervals_specialbehav(derivatives_base: Path):
    """
    Finds the start and stop time for each trial, so that we can restrict spike times to for each goal
    NOTE Is based on the labview data
    Args:
        rawsession_folder: path to rawsession folder

    Raises:
        FileNotFoundError: No file for today_alltrials found
    """
    print("Creating intervals special behaviour")
    
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Add trial length to df
    path = Path(derivatives_base) / "metadata"/ "trials_length.csv"
    df_length = pd.read_csv(path)
    lengths = df_length['trial length (s)'].to_numpy()

    # Df with restrictions
    try: 
        path =rawsession_folder/"task_metadata"/"restricted_df.xlsx"
        df_restricted = pd.read_excel(path)
    except:
        path = rawsession_folder/"task_metadata"/"restricted_df.csv"
        df_restricted = pd.read_csv(path)
        

    # Getting the cumulative length
    cumul_length = [0]
    length_so_far = 0
    
    for i in range(len(lengths) - 1):
        length_so_far += lengths[i]
        cumul_length.append(length_so_far)
   
    df_restricted = df_restricted.add(cumul_length, axis = 0)

    output_path = rawsession_folder/"task_metadata"/"restricted_final.csv"
    df_restricted.to_csv(output_path, index = False)
    print(f"Saved data to {output_path}")

def check_restricted_df_exists(derivatives_base: Path) -> bool:
    """ Returns whether restricted df exists"""
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Add trial length to df
    path = rawsession_folder/"task_metadata"/"restricted_df.csv"
    return path.exists()

def make_restricted_df(derivatives_base: Path, goals_to_include: list, trials_to_include: list) -> None:
    """ Makes a basic version of restricted df. columns: start g1 wrong	end g1 wrong	start g1	end g1	start g2	end g2
    Number of rows = len(trials_to_include)
    This assumes that 0 is not in goals_to_include, because that data has to be inputted by the experimentalist
    first three columns will have only zeros
    end g1, start g2, and end g2 are read from the alltrials csv file
    
    
    Inputs:
        derivatives_base: path to derivatives folder
        goals_to_include: which goals are in the trial, from [0,1,2]
        trials_to_include: trial numbers
    
    """
    print("Making restricted df")
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    # DF info
    col_names = ["start g1 wrong", "end g1 wrong", "start g1", "end g1", "start g2", "end g2"]
    num_rows = len(trials_to_include)
    zero_rows = np.zeros(num_rows)
    
    # Getting start and end times
    alltrials_path = rawsession_folder/"behaviour"/"alltrials_trialday.csv"
    alltrials_df = pd.read_csv(alltrials_path)
    
    end_g1 = alltrials_df["Goal 1 end"].to_numpy()
    
    if 2 in goals_to_include:
        start_g2 = alltrials_df["Goal 2 start"].to_numpy()
        end_g2 = alltrials_df["Trial duration"].to_numpy()
        data = np.vstack([zero_rows, zero_rows, zero_rows, end_g1, start_g2, end_g2]).T
        breakpoint()
    else:
        breakpoint()
        data = np.vstack([zero_rows, zero_rows, zero_rows, end_g1, zero_rows, zero_rows]).T
        
    df = pd.DataFrame(data, columns=col_names)
    output_path = rawsession_folder/"task_metadata"/"restricted_df.csv"
    df.to_csv(output_path, index = False)
    
    
if __name__ == "__main__":
    rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-002_id-1R\ses-01_date-10092025"
    #rawsession_folder = r"S:\Honeycomb_maze_task\rawdata\sub-003_id-2F\ses-01_date-17092025"
    create_intervals_specialbehav(rawsession_folder)