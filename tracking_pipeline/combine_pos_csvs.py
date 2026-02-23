import numpy as np
import pandas as pd
from preprocessing.get_length_all_trials import get_length_all_trials
from pathlib import Path
from typing import Literal

BodyPart = Literal["ears", "center"]

def combine_pos_csvs(derivatives_base: Path, trials_to_include: list, frame_rate: int= 25):
    r"""g
    Combines all data from XY_HD_t{tr}.csv (for tr in trials_to_include) into one csv called HD_XY_alltrials.csv
    and saves it in the same folder as the XY_HD_t{tr}.csvss

    Adds a bit of padding to match the trial length
    
    Input:
        derivatives_base: path to derivatives folder
        trials_to_include: our trial numbers
        frame_rate (int: 25): frame rate of video
    
    Raises:
    Exception: 
        - If path to XY/HD data folder does not exist
        - If XY data file for a given trial is missing
        - If 'trials_length.csv' is missing and cannot be generated
    ValueError:
        - If positional data for a trial is longer than recording data
        - If padding lengths between ears and center data differ
    AssertionError:
        - If total concatenated duration differs from expected by > 2 seconds
        
    Outputs:
        derivatives_base\analysis\spatial_behav_data\XY_and_HD\XY_HD_alltrials.csv: all the trials concatenated together with padding inbetween
        trials to match trial length. xy position corresponds to center of ears.
        derivatives_base\analysis\spatial_behav_data\XY_and_HD\XY_HD_alltrials_center.csv: same as before but xy is the center of the animal

    """
    folder_path, trials_length = get_folder(derivatives_base, trials_to_include)

    data = {"x": [], "y": [], "hd": [], "speed": []}
    df = pd.DataFrame(data)
    df_center = pd.DataFrame(data) # This will store the coordinates with the center positions

    total_length_s = 0 # total trial length in seconds
    
    # Going over trials
    for tr in trials_to_include:
        # Getting length for this trial in frames
        length_fr, total_length_s = get_trial_length(trials_length, total_length_s, tr, frame_rate)

        # Df head/ears
        df_tr = get_df_trial(folder_path, tr, length_fr, method = "ears") # load df for this trial (df_tr)
        padding = get_padding(df_tr, length_fr) # get padding
        df = pd.concat([df, df_tr, padding]) # add df_tr + padding to df
        
        # Df center
        df_tr_center = get_df_trial(folder_path, tr, length_fr, method = "center")
        padding_cr = get_padding(df_tr_center, length_fr)
        df_center = pd.concat([df_center, df_tr_center, padding_cr])

        # Trouble shooting
        if len(padding) != len(padding_cr):
            raise ValueError("len(padding) != len(padding_cr). verify files are the same length")

    #  Saves df to folder_path/XY_HD_alltrials.csv (or _center.csv if method == center)
    save_df(df, folder_path, method = "ears")
    save_df(df_center, folder_path, method = "center")

    # Verifies whether length of df matches the total length of the trial
    check_difference_df(df, total_length_s, frame_rate)
    
def get_folder(derivatives_base: Path, trials_to_include: list) -> tuple[Path, pd.DataFrame]:
    """ Gets data needed for session"""
    # rawsession folder
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent

    # Input folder
    folder_path = derivatives_base/"analysis"/"spatial_behav_data"/"XY_and_HD"
    if not folder_path.exists():
        raise Exception("Path to XY and HD data does not exist")
    
    # trials length dataframe
    trials_length_csv = derivatives_base/'metadata'/'trials_length.csv'
    if not trials_length_csv.exists():
        get_length_all_trials(derivatives_base, trials_to_include)
    trials_length = pd.read_csv(trials_length_csv)
    print("Using trials length file")
    print(trials_length)
    
    return folder_path, trials_length
   
def get_trial_length(trials_length: pd.DataFrame, total_length_s: float, tr:int, frame_rate: int = 25) -> tuple[float, float]:
    """ Get the trial length in frames and the total trials length in seconds"""
    trials_length_tr = trials_length[trials_length["trialnumber"] == tr] # row for this trial
    length_s = trials_length_tr.iloc[0,2] # Length is in third column
    length_fr=np.round( length_s * frame_rate).astype(np.int32)
    total_length_s += length_s 
    return length_fr, total_length_s

def get_df_trial(folder_path: Path, tr: int, length_fr: int, method: BodyPart) -> pd.DataFrame:
    """ Loads positional df of the trial"""
    if method == "ears":
        input_path = folder_path/f"XY_HD_t{tr}.csv"
    elif method == "center":
        input_path = folder_path/f"XY_HD_center_t{tr}.csv"
    if not input_path.exists():
        raise Exception(f"Path to XY data for trial {tr} not found")
    
    df_tr = pd.read_csv(input_path)

    if len(df_tr) > length_fr:
        df_tr = df_tr.iloc[:length_fr].copy() # This is done for when video is longer than neural recording
        print(f"Cropping df_tr for trial {tr}")
        #raise ValueError(f"Positional data for trial {tr} is longer than recording data. Fix error.")
    
    return df_tr

def get_padding(df_tr: pd.DataFrame, length_fr: int) -> pd.DataFrame:
    """ Returns padding, which is a nan df of length length_fr - len(df_tr)"""
    padding_len = length_fr - len(df_tr) # This is how much longer the recording session was than the actual trial. 
    # We add extra lines to the positional data (with nan values) in order to match the length
    nan_rows = np.repeat(np.nan, padding_len)
    padding = pd.DataFrame({"x": nan_rows, "y": nan_rows, 'hd': nan_rows})
    return padding

def save_df(df: pd.DataFrame, folder_path: Path, method: BodyPart) -> None:
    """ Saves df to folder_path/XY_HD_alltrials.csv (or _center.csv if method == center)"""
    if method=="ears":
        output_path = folder_path/"XY_HD_alltrials.csv"
    elif method =="center":
        output_path = folder_path/"XY_HD_alltrials_center.csv"
        
    df.to_csv(output_path, index = False)
    print(f"Dataframe saved to {output_path}")

def check_difference_df(df: pd.DataFrame, total_length_s: float,  frame_rate: int = 25):
    """ Verifies whether length of df matches the total length of the trial"""
    len_df_s = len(df)/frame_rate
    diff_s = total_length_s - len_df_s
    print(f"Difference between concat df length and total trial length: {diff_s:.2f} s")
    if diff_s > 2:
        raise AssertionError("Difference between concat df length and total trial length is greater than 2 seconds. Verify where error occurs")


if __name__ == "__main__":
    trials_to_include = np.arange(1,14)
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    combine_pos_csvs(derivatives_base, trials_to_include)