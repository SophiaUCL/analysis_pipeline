import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def restrict_spiketrain_specialbehav(spike_train: list, rawsession_folder: Path, goal: int):
    """
    Restricts the spike train to the intervals of the goal.
    Note! Assumtes the spike train is IN SECONDS

    Args:
        spike_train (array): spike times of cell in seconds
        rawsession_folder (str): path to the raw session folder
        goal (int, optional): goal number (0,1, 2) 0 meaning that it is rat going to g2 during g1. 

    Returns:
        spike times in seconds but only within the intervals
    """
    if goal not in (0, 1,2):
        raise ValueError("Goal must be 0, 1 or 2") 
    path = rawsession_folder/"task_metadata"/"restricted_final.csv"

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

def get_spike_train(sorting, sample_rate, rawsession_folder, unit_id, g, frame_rate, pos_data):
    """ Gets the spiketrain for unit unit_id for goal g. Spiketrain returned is in frames
    If g < 3 (so g0, 1 or 2), we restrict the spiketrain to only that goal
    g == 3 is full trial
    g == 4 is only open field
    """

    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_secs = spike_train_unscaled / sample_rate  # trial data in seconds

    # If we're only looking at one goal, restrict the spiketrain to match only period of goal 1 or 2
    if g < 3:
        spike_train_restricted = restrict_spiketrain_specialbehav(spike_train_secs, rawsession_folder=rawsession_folder,
                                                                  goal=g)
    elif g == 3:
        spike_train_restricted = spike_train_secs
    elif g == 4:
        # Restrict to open field. Assumes that the open field trial is the last one!
        open_field_start = get_openfield_starttime(rawsession_folder)
        spike_train_restricted =np.array([el for el in spike_train_secs if el > open_field_start])

    spike_train = spike_train_restricted * frame_rate
    spike_train = [np.int32(np.round(el)) for el in spike_train if el < len(pos_data) - 1]

    return spike_train

def get_openfield_starttime(rawsession_folder: Path):
    """ Gets start time of open field trial by finding the cumulative length of the previous trials"""
    trials_length_path = rawsession_folder/"task_metadata"/"trials_length.csv"
    df = pd.read_csv(trials_length_path)
    trials_length = df.iloc[:,2].to_numpy()
    trials_length = trials_length[:-1] # Remove last element    
    start_openfield = sum(trials_length)
    return start_openfield
