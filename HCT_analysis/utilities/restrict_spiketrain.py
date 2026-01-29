import numpy as np
import os
import pandas as pd
from HCT_analysis.utilities.create_intervals import create_intervals_df

def restrict_spiketrain(spike_train, rawsession_folder, goal: int):
    """
    Restricts the spike train to the intervals of the goal.
    Note! Assumtes the spike train is IN SECONDS

    Args:
        spike_train (array): spike times of cell in seconds
        rawsession_folder (str): path to the raw session folder
        goal (int, optional): goal number (1 or 2)

    Returns:
        spike times in seconds but only within the intervals
    """
    if goal not in (1,2):
        raise ValueError("Goal must be 1 or 2") 
    path = os.path.join(rawsession_folder, "task_metadata", f"goal_{goal}_intervals.csv")
    
    if not os.path.exists(path):
        create_intervals_df(rawsession_folder)
    intervals_df = pd.read_csv(path)

    # Convert to list of tuples
    intervals = list(zip(intervals_df['start_time'], intervals_df['end_time']))

    # Restrict
    mask = np.zeros_like(spike_train, dtype=bool)
    for start, end in intervals:
        mask |= (spike_train >= start) & (spike_train <= end)

    return spike_train[mask]