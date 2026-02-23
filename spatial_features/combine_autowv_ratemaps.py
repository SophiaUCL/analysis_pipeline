
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Literal
import spikeinterface.extractors as se
from pathlib import Path
from spatial_features.utils.spatial_features_utils import load_unit_ids

UnitTypes = Literal['pyramidal', 'good', 'all']

def combine_autowv_ratemaps(derivatives_base: Path, unit_type: UnitTypes, rmap_per_goal: bool = False) -> None:
    """
    Combines:
      - left: autocorrelogram + waveform
      - right-top: spike count over trials
      - right-bottom: ratemap + head direction
    
    All images are scaled to have the same width on the right column and total equal height.
    
    Inputs
    --------
    derivatives_base (Path): Path to derivatives folder
    unit_type (pyramidal, all, or good): Which units are used to combine the plots
    rmap_per_goal (bool: False): if true, loads the ratemaps per goal
    
    Saves
    -------
      analysis/cell_characteristics/spatial_features/autowv_ratemap_combined/unit_{unit}.png
    s
    Called by
    --------
    spatial_processing_pipeline
    """
    
    # Load data files
    kilosort_output_path = derivatives_base/ 'ephys'/"concat_run"/"sorting"/"sorter_output"
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    # output paths
    if not rmap_per_goal:
        extension = "autowv_ratemap_combined"
    else:
        extension =  "autowv_ratemap_pergoal_combined"
    output_folder =derivatives_base/"analysis"/"cell_characteristics"/"spatial_features"/extension
    output_folder.mkdir(exist_ok = True)
       
    # getting unit ids
    unit_ids = sorting.unit_ids
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    
            
    print("Combining autocorrelogram + waveform plots with ratemap + hd plots")
    print(f"Path to output folder: {output_folder}")
    for unit_id in tqdm(unit_ids):
        try:
            left_path = derivatives_base / "analysis"/"cell_characteristics"/"unit_features"/"auto_and_wv"/f"unit_{unit_id:03d}.png"
            if not rmap_per_goal:
                folder = "ratemaps_and_hd"
            else:
                folder = "ratemaps_and_hd_allgoals"
            b_right_path =   derivatives_base / "analysis"/"cell_characteristics"/"spatial_features"/folder/f"unit_{unit_id}_rm_hd.png"
            t_right_path = derivatives_base / "analysis"/"cell_characteristics"/"unit_features"/"spikecount_over_trials"/f"unit_{unit_id}_sc_over_trials.png"
            # Open images
            left = Image.open(left_path)
            b_right = Image.open(b_right_path)
            t_right = Image.open(t_right_path)
        except:
            print(f"Could not find images for unit {unit_id}, skipping.")
            continue

        
        if rmap_per_goal:
            scale = (b_right.height + t_right.height)/left.height
            new_left_width = int(left.width * scale)

            left = left.resize(
                (new_left_width, b_right.height + t_right.height),
                Image.LANCZOS
            )
        

        total_width = left.width + np.max([b_right.width, t_right.width])
        total_height = np.max([left.height, b_right.height + t_right.height])

        # Create a blank white canvas
        combined = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

        # Paste the images
        combined.paste(left, (0, 0))  # left side
        combined.paste(t_right, (left.width,b_right.height))  # right side
        combined.paste(b_right, (left.width,0)) # right side

        # Save the combined result
        output_file = output_folder /  f"unit_{unit_id:03d}.png"
        combined.save(output_file)




def combine_autowv_ratemaps_vectorfields(derivatives_base: Path, unit_type: UnitTypes) -> None:
    """
    Combines:
      - left: autocorrelogram + waveform
      - right-top: spike count over trials
      - right-bottom: ratemap + head direction
      - right: vectorfields
    
    Inputs
    --------
    derivatives_base (Path): Path to derivatives folder
    unit_type (pyramidal, all, or good): Which units are used to combine the plots
    
    All images are scaled to have the same width on the right column and total equal height.
    
    Saves to
    --------
      analysis/cell_characteristics/spatial_features/autowv_ratemap_vectorfields_combined/unit_{unit}.png
    
    Called by
    --------
    spatial_processing_pipeline
    """
    # Load data files
    kilosort_output_path = derivatives_base/ 'ephys'/"concat_run"/"sorting"/"sorter_output"
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
    # output paths
    output_folder =derivatives_base/"analysis"/"cell_characteristics"/"spatial_features"/"autowv_ratemap_vectorfields_combined"
    output_folder.mkdir(exist_ok = True)
       
    # getting unit ids
    unit_ids = sorting.unit_ids
    unit_ids = load_unit_ids(derivatives_base, unit_type, unit_ids)
    
            
    print("Combining autocorrelogram + waveform plots with ratemap + hd plots")
    print(f"Path to output folder: {output_folder}")
    for unit_id in tqdm(unit_ids):
        try:
            left_path = derivatives_base / "analysis"/"cell_characteristics"/"unit_features"/"auto_and_wv"/f"unit_{unit_id:03d}.png"
            b_right_path =   derivatives_base / "analysis"/"cell_characteristics"/"spatial_features"/"ratemaps_and_hd"/f"unit_{unit_id}_rm_hd.png"
            t_right_path = derivatives_base / "analysis"/"cell_characteristics"/"unit_features"/"spikecount_over_trials"/f"unit_{unit_id}_sc_over_trials.png"
            right_path =derivatives_base / "analysis"/"cell_characteristics"/"spatial_features"/"vector_fields"/f"vector_fields_unit_{unit_id}.png"
            # Open images
            left = Image.open(left_path)
            b_right = Image.open(b_right_path)
            t_right = Image.open(t_right_path)
            right = Image.open(right_path)
        except:
            print(f"Could not find images for unit {unit_id}, skipping.")
            continue


        scale = left.height / right.height
        new_right_width = int(right.width * scale)

        right = right.resize(
            (new_right_width, left.height),
            Image.LANCZOS
        )
        total_width = left.width + np.max([b_right.width, t_right.width]) + right.width
        total_height = np.max([left.height, b_right.height + t_right.height, right.height])

        # Create a blank white canvas
        combined = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

        # Paste the images
        combined.paste(left, (0, 0))  # left side
        combined.paste(t_right, (left.width,b_right.height))  # right side
        combined.paste(b_right, (left.width,0)) # right side
        combined.paste(right, (t_right.width + left.width, 0))

        # Save the combined result
        output_file = output_folder /  f"unit_{unit_id:03d}.png"
        combined.save(output_file)

