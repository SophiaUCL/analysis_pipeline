import numpy as np
import os
import glob
from pathlib import Path

def find_paths(base_path: Path, subject_number: str, session_number: str, trial_session_name: str):
    """
    Finds the paths to the rawsession folder and derivatives folder
    
    Args:
        base_path (Path): path to the folder with the data (for example, Z:\Eylon\Data\Spatiotemporal_Task)
        subject_number (str): subject number (for example, 002)
        session_number (str): session number (for example, 01)
        trial_session_name (str): trial session name (for example, all_trials)

    Raises:
        FileNotFoundError: If the required folders are not found

    Returns:
        rawsession_folder (Path): path to the raw session folder
        derivatives_base (Path): path to the derivatives folder
        rawsubject_folder (Path): path to the raw subject folder
        session_name (str): name of the session
    
    Called by:
        main_pipeline.py
    """
    matching_folders= list(base_path.glob(rf"rawdata\sub-{subject_number}*"))
    if matching_folders:
        subject_folder = matching_folders[0]  
        print(f"Subject folder:         {subject_folder}")
    else:
        raise FileNotFoundError(f"No subject folder found for pattern {rf"rawdata\sub-{subject_number}*"}")
    rawsubject_folder = subject_folder  # Full path to the subject folder

    # === Session
    matching_sessions= list(subject_folder.glob(rf"ses-{session_number}*"))

    if matching_sessions:
        session_name = matching_sessions[0].name
        rawsession_folder = matching_sessions[0]  # Full path
        print(f"Session folder:         {rawsession_folder}")
    else:
        raise FileNotFoundError(f"No session folder found for pattern {rf"ses-{session_number}*"}")
    
    # === Derivates folder
    subject_name = subject_folder.name

    derivatives_base = base_path / 'derivatives' / subject_name / session_name/ trial_session_name
    derivatives_base.mkdir(parents = True, exist_ok=True)

    print("Derivatives folder:      ", derivatives_base)

    return  Path(derivatives_base), Path(rawsession_folder), Path(rawsubject_folder), session_name