import numpy as np
import pandas as pd
import re
from pathlib import Path

def get_length_all_trials(derivatives_base: Path, trials_to_include: list[int]) -> None:
    """
    Creates a file with the trial length for all the trials (which it obtains from the meta files)
    Saves it to derivatives_base/metadata/trials_length.csv
    
    Inputs
    -----------
    derivatives_base (Path): path to derivatives base folder
    trials_to_include (list[int]): trial numbers
        
    Outputs
    -------------
    derivatives_base/metadata/trials_length.csv
        csv file with trial numbers, g numbers, trial length (s), and cumulative length (length of all trials up until that trial)
        
    Raises
    ------------
    raise ValueError(f"Error in g numbers, order is wrong. Do zero padding. g_numbers = {g_numbers}")
        if g numbers are wrongly ordered (this happens if the g numbers are not zero padded beforehand, then g10 comes before g9)
        

    Called by
    -----------
    main_pipeline.py
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    # Loading data paths
    ephys_path = rawsession_folder /  'ephys'
    output_folder = derivatives_base / "metadata"
    output_folder.mkdir(exist_ok = True)
    output_path = output_folder / "trials_length.csv"


    # Step 1: First we find all run folders (e.g., ses-01_g0, ses-01_g1, etc.)
    run_folders = list(ephys_path.glob(r"ses*"))

    print(f"Found {len(run_folders)} run fo  lder(s) in {ephys_path}:\n")

    # going over all the folders
    for folder in run_folders:
        base = folder.name
        dir_parent = folder.parent

        match = re.search(
            r'(?P<session>ses[-_]\d+).*?_g(?P<group>\d+)$',
            base
        )

        if match:
            new_name = f"{match['session']}_g{int(match['group']):02d}"
            new_path = dir_parent / new_name

            if new_path != folder:
                print(f"Renaming: {base} → {new_name}")
                folder.rename(new_path)
        else:
            print(f"Skipping {base}: no match for 'g' number pattern.")


    # Assinging new folders
    run_folders = list(ephys_path.glob(r"ses*"))

    g_numbers = []
    trials_length = []
    cumul_length = []
    cumul_length_num = 0
    # Step 2: Process each run folder
    for run_folder in run_folders:

        basename = run_folder.name
        match = re.search(r'ses[-_]\d+.*?_g(\d+)', basename)
        if match:
            group_number = int(match.group(1))
            g_numbers.append(group_number)
            if len(g_numbers) > 1 and g_numbers[-2] > g_numbers[-1]:
                raise ValueError(f"Error in g numbers, order is wrong. Do zero padding. g_numbers = {g_numbers}")

        else:
            print(f"  Warning: Could not extract group number from {basename}")

            
        # Step 2a: Get subfolder inside run_folder (e.g., ses-01_g0_imec0)
        subfolders = [f for f in run_folder.iterdir() if f.is_dir()]
        if not subfolders:
            print(f"  Warning: No subfolders found in {run_folder}. Skipping.")
            continue

        subfolder_name = subfolders[0]
        subfolder_path = run_folder / subfolder_name

        # Step 2b: Look for meta file
        meta_matches = list(subfolder_path.glob("*meta*"))
        if not meta_matches:
            print(f"  Warning: No meta files found in {subfolder_path}. Skipping.")
            continue
        if len(meta_matches) > 1:
            print(f"  Warning: Multiple meta files found. Using the first one.")

        meta_path = meta_matches[0]

        # Step 2c: Parse meta file to get fileTimeSecs
        try:
            with open(meta_path, 'r') as f:
                content = f.read()

            match = re.search(r'fileTimeSecs\s*=\s*([\d.]+)', content)
            if match:
                file_time_secs = float(match.group(1))
                print(f"  Found fileTimeSecs = {file_time_secs}")
            else:
                print(f"  Warning: 'fileTimeSecs' not found in {meta_path}")
            trials_length.append(file_time_secs)
            cumul_length.append(cumul_length_num)
            cumul_length_num += file_time_secs
        except Exception as e:
            print(f"  Error reading meta file {meta_path}: {e}")
    data = {"trialnumber": trials_to_include, "g": g_numbers, "trial length (s)": trials_length, "cumulative length": cumul_length}
    trial_length_df = pd.DataFrame(data)
    trial_length_df.to_csv(output_path, index = False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    rawsession_folder = r"E:\Honeycomb_task_1g\rawdata\sub-001_id-2H\ses-01_date-01282026"
    trials_to_include = np.arange(1,27)
    get_length_all_trials(rawsession_folder, trials_to_include)