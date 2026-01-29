# %%
import numpy as np
import os
import glob
import pandas as pd
import re


def get_length_all_trials(rawsession_folder, trials_to_include):
    """
    Creates a file with the trial length for all the trials
    Saves it to rawsession_folder/task_metadata/trials_length.csv
    
    Inputs:
        rawsession_folder: path to rawsession folder
        trials_to_include: trial numbers
    """
    # Loading data paths
    ephys_path = os.path.join(rawsession_folder, 'ephys')
    output_folder = os.path.join(rawsession_folder, "task_metadata")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, "trials_length.csv")


    # Step 1: First we find all run folders (e.g., ses-01_g0, ses-01_g1, etc.)
    pattern = os.path.join(ephys_path, "ses*")
    run_folders = [folder for folder in glob.glob(pattern) if os.path.isdir(folder)]

    print(f"Found {len(run_folders)} run folder(s) in {ephys_path}:\n")

    # going over all the folders
    for folder in run_folders:
        base = os.path.basename(folder)
        dir_parent = os.path.dirname(folder)

        match = re.search(
            r'(?P<session>ses[-_]\d+).*?_g(?P<group>\d+)$',
            base
        )

        if match:
            new_name = f"{match['session']}_g{int(match['group']):02d}"
            new_path = os.path.join(dir_parent, new_name)

            if new_path != folder:
                print(f"Renaming: {base} → {new_name}")
                os.rename(folder, new_path)
        else:
            print(f"Skipping {base}: no match for 'g' number pattern.")


    # Assinging new folders
    run_folders = [folder for folder in glob.glob(pattern) if os.path.isdir(folder)]

    g_numbers = []
    trials_length = []
    
    # Step 2: Process each run folder
    for run_folder in run_folders:

        basename = os.path.basename(run_folder)
        match = re.search(r'ses[-_]\d+.*?_g(\d+)', basename)
        if match:
            group_number = int(match.group(1))
            g_numbers.append(group_number)
            if len(g_numbers) > 1 and g_numbers[-2] > g_numbers[-1]:
                raise ValueError(f"Error in g numbers, order is wrong. Do zero padding. g_numbers = {g_numbers}")

        else:
            print(f"  Warning: Could not extract group number from {basename}")

            
        # Step 2a: Get subfolder inside run_folder (e.g., ses-01_g0_imec0)
        subfolders = [f for f in os.listdir(run_folder) if os.path.isdir(os.path.join(run_folder, f))]
        if not subfolders:
            print(f"  Warning: No subfolders found in {run_folder}. Skipping.")
            continue

        subfolder_name = subfolders[0]
        subfolder_path = os.path.join(run_folder, subfolder_name)

        # Step 2b: Look for meta file
        meta_pattern = os.path.join(subfolder_path, "*meta*")
        meta_matches = glob.glob(meta_pattern)

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
        except Exception as e:
            print(f"  Error reading meta file {meta_path}: {e}")

    data = {"trialnumber": trials_to_include, "g": g_numbers, "trial length (s)": trials_length}
    trial_length_df = pd.DataFrame(data)
    trial_length_df.to_csv(output_path, index = False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    rawsession_folder = r"E:\Honeycomb_task_1g\rawdata\sub-001_id-2H\ses-01_date-01282026"
    trials_to_include = np.arange(1,27)
    get_length_all_trials(rawsession_folder, trials_to_include)