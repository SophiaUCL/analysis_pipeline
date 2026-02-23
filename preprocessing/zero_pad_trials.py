import re
from pathlib import Path
def zero_pad_trials(rawsession_folder: Path) -> None:
    """
    Renames folders in ephys folder from g1, g2, g10 to g01, g02, g10 (0 pads them)

    Inputs
    --------
    rawsession_folder (Path): path to rawsession_folder
    
    
    Called by
    --------
    main_pipeline.py
    """
    # Loading folders
    ephys_path = rawsession_folder / "ephys"
    run_folders = [folder for folder in list(ephys_path.glob(r"ses*")) if folder.is_dir()]

    print(f"Found {len(run_folders)} run folder(s) in {ephys_path}:\n")

    # GOING OVER FOLDERS
    for folder in run_folders:
        base = folder.name
        dir_parent = folder.parent

        match = re.search(r'(ses.*_g)(\d+)$', base)

        if match:
            prefix, num = match.groups()
            new_name = f"{prefix}{int(num):02d}"  # pad to 2 digits (e.g. g00, g01, g10)
            new_path = dir_parent / new_name

            if new_path != folder:
                print(f"Renaming: {base} → {new_name}")
                folder.rename( new_path)
        else:
            print(f"Skipping {base}: no match for 'g' number pattern.")