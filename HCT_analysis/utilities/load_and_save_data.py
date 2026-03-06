import pickle
import os

import sys

""" Jake's code to save and load data"""

def load_pickle(filename, dir):
    """
    Load a pickle file.
    
    Parameters
    ----------
    filename : str
        The filename of the pickle file.
    dir : str
        The directory of the pickle file.
        
    Returns
    -------
    data : object
        The data from the pickle file.
    """
    # find the file in the directory
    files = os.listdir(dir)

    # remove any entries in files that aren't pickle files
    files = [f for f in files if f.endswith('.pickle') or f.endswith('.pkl')]

 
    # find the file that matches the filename
    
    for f in files:

        # f2 is f without the extension
        f2 = f.split('.')[0]

        if filename == f2:
            filename = f
            break

    filepath = os.path.join(dir, filename)
    with open(os.path.join(dir, filepath), 'rb') as handle:
        data = pickle.load(handle)
    return data

def save_pickle(data, filename, dir):
    """
    Save a pickle file.
    
    Parameters
    ----------
    filename : str
        The filename of the pickle file.
    dir : str
        The directory of the pickle file.
        
    Returns
    -------
    data : object
        The data from the pickle file.
    """
    # if filename doesn't include pickle extension, add it
    if not filename.endswith('.pickle') or not filename.endswith('.pkl'):
        filename = filename + '.pickle' 
    
    filepath = os.path.join(dir, filename)
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)