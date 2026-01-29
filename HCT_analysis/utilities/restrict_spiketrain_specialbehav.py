import numpy as np
import os
import pandas as pd


def restrict_spiketrain_specialbehav(spike_train, rawsession_folder, goal: int):
    """
    Restricts the spike train to the intervals of the goal.
    Note! Assumtes the spike train is IN SECONDS

    Args:
        spike_train (array): spike times of cell in seconds
        rawsession_folder (str): path to the raw session folder
        goal (int, optional): goal number (0,1 or 2) 0 meaning that it is rat going to g2 during g1

    Returns:
        spike times in seconds but only within the intervals
    """
    if goal not in (0, 1,2):
        raise ValueError("Goal must be 0, 1 or 2") 
    path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")
    intervals_df = pd.read_csv(path)
   
    start_col = 2*goal
    end_col = 2*goal + 1
    
    # get only start and end col
    intervals_df_restr = intervals_df.iloc[:, start_col:end_col + 1]
    
    # Convert to list of tuples
    intervals = list(zip(intervals_df_restr.iloc[:,0], intervals_df_restr.iloc[:,1]))
    
    # Restrict
    mask = np.zeros_like(spike_train, dtype=bool)
    for start, end in intervals:
        mask |= (spike_train > start) & (spike_train < end)
            
        
        

    return spike_train[mask]