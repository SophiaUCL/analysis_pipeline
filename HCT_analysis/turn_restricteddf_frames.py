import os
import pandas as pd
import numpy as np
from pathlib import Path

def turn_restricteddf_frames(derivatives_base: Path, frame_rate: int = 25):
    """ Converts restricted df columns from seconds to frames and saves it
    
    Inputs
    -----
    derivatives_base (Path): path to derivatives folder
    frame_rate (int: 25): frame rate of the video
    
    
    Outputs
    ------
    rawsession_folder/"task_metadata"/"restricted_df_frames.csv": restricted df but then in frames
    """
    
    # rawsession folder
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "raw_data")).parent
    
    # get file
    path = rawsession_folder/"task_metadata"/'restricted_final.csv'
    df = pd.read_csv(path)
    
    # make into frames
    df = np.round(df*frame_rate).astype(int) 
    
    # export
    output_folder = rawsession_folder/"task_metadata"/"restricted_df_frames.csv"
    df.to_csv(output_folder, index = False)
    
    print(f"Saved df to {output_folder}")
    
if __name__ == "__main__":
    derivatives_base =  r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    turn_restricteddf_frames(derivatives_base)
