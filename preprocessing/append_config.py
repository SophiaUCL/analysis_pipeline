import json
from pathlib import Path


def append_config(derivatives_base: Path, new_data: dict, filename: str ="config.json"):
    """
    Appends data to the config file in the derivatives folder

    Inputs
    -----
    derivatives_base: Path 
        path to derivatives_base
    data: dict
        Dictionary with configuration data
    filename: str, defaults to "config.json"
        name used to save the data under
        

    Saves
    -----
    derivatives_base/filename
        new data added to the existing data in the json file
        
    Called by
    -----
    main_pipeline.py
    """
    config_file = derivatives_base / filename

    if config_file.exists():
        with open(config_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # merge deeply instead of replacing
    deep_update(existing_data, new_data)

    with open(config_file, "w") as f:
        json.dump(existing_data, f, indent=4)
        
        
def deep_update(d, u):
    """Recursively update dict d with dict u"""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


