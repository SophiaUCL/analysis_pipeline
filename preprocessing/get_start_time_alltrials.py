import numpy as np
import pandas as pd
import re
from pathlib import Path

def get_start_time_alltrials(derivatives_base: Path):
    """
    Creates a file with the trial start times for all the trials (which it obtains from the meta files)
    Saves it to rawsession_folder/task_metadata/trials_start_times.csv
    
    Inputs
    -----------
    derivatives_folder (Path): Path to derivatives folder
        
    Outputs
    -------------
    rawsession_folder/task_metadata/trials_start_times.csv
        csv file with trial numbers, g numbers, and start times (HH:MM:SS)
        

    Called by
    -----------
    No other file
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    # Loading data paths
    ephys_path = rawsession_folder /  'ephys'
    output_folder = rawsession_folder / "task_metadata"
    output_folder.mkdir(exist_ok = True)
    output_path = output_folder / "trials_start_times.csv"


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
    trials_start_times = []
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

            match = re.search(r'fileCreateTime\s*=\s*([^\n]+)', content)

            if match:
                full_datetime = match.group(1).strip()  # 2026-02-12T14:44:10
                
                # Split at "T"
                if "T" in full_datetime:
                    time_part = full_datetime.split("T")[1]
                    print(f"  Found start time = {time_part}")
                    trials_start_times.append(time_part)
                else:
                    print(f"  Warning: Unexpected datetime format in {meta_path}")
            else:
                print(f"  Warning: 'fileCreateTime' not found in {meta_path}")

        except Exception as e:
            print(f"  Error reading meta file {meta_path}: {e}")

    data = {"g": g_numbers, "trial start time (s)": trials_start_times}
    trial_starts_df = pd.DataFrame(data)
    trial_starts_df.to_csv(output_path, index = False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-02_date-12022026\first_run_1302"
    get_start_time_alltrials(Path(derivatives_base))