
import os
import glob
import re

def zero_pad_trials(rawsession_folder):
    """
    Renames folders in ephys folder from g1, g2, g10 to g01, g02, g10 (0 pads them)

    Args:
        rawsession_folder: path to rawsession_folder
    """
    # Loading folders
    ephys_path = os.path.join(rawsession_folder, "ephys")
    pattern = os.path.join(ephys_path, "ses*")
    run_folders = [folder for folder in glob.glob(pattern) if os.path.isdir(folder)]

    print(f"Found {len(run_folders)} run folder(s) in {ephys_path}:\n")

    # gOING OVER FOLDERS
    for folder in run_folders:
        base = os.path.basename(folder)
        dir_parent = os.path.dirname(folder)

        match = re.search(r'(ses.*_g)(\d+)$', base)

        if match:
            prefix, num = match.groups()
            new_name = f"{prefix}{int(num):02d}"  # pad to 2 digits (e.g. g00, g01, g10)
            new_path = os.path.join(dir_parent, new_name)

            if new_path != folder:
                print(f"Renaming: {base} → {new_name}")
                os.rename(folder, new_path)
        else:
            print(f"Skipping {base}: no match for 'g' number pattern.")