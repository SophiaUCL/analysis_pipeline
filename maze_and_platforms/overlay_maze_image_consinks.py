import numpy as np
from pathlib import Path
from typing import Literal
from maze_and_platforms.find_platforms import (
    hex_grid,
    calculate_cartesian_coords,
    translate_coords,
    save_params,
    get_image,
    get_params,
    makefig,
)
import json

OverlayMethod = Literal["video", "image"]

def overlay_maze_image_consinks(derivatives_base: Path, 
                       method: OverlayMethod,
                       )-> tuple[bool, np.ndarray]:
    """
    This code overlays a hexagonal grid on
    the image in the code files that has the maze with allplatforms up (if method == image)
    the first frame of the videos for this session (if method == video)
    In the code files, there's an image saved where all maze platforms are up. This function overlays the hex grid
    on top of that image to show the maze layout
    
    Inputs:
    derivatives_base: path to derivatives folder
    method: video or image, whether to obtain the image to overlay from the video recordings (video) or from the image saved in the code (image)
    Outputs:
        Into derivatives/analysis/maze_overlay:
            maze_overlay_consinks.png: image with hex grid overlay
            maze_overlay_params_consinks.json: parameters used to create the overlay
    
    """
    # Loading rawsession folder
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    # Obtaining image used for overlay
    img = get_image(rawsession_folder, method)

    # Output folder
    output_path = derivatives_base / "analysis" / "maze_overlay" / "maze_overlay_consinks.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    params_path = derivatives_base / "analysis" / "maze_overlay" / "maze_overlay_params.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    desired_x = params['x_center']
    desired_y = params['y_center']
    hex_side_length = params['hex_side_length']
    rotation = params['rotation']
    theta = params['theta']
    radius = 6
    
    # Calculate initial Cartesian coordinates
    coord = hex_grid(radius) # coordinates
    hcoord2, vcoord2 = calculate_cartesian_coords(coord, hex_side_length)
    hcoord_translated, vcoord_translated = translate_coords(hcoord2, vcoord2, theta, desired_x, desired_y, num_plats = 127)
    # plots figure
    good_overlay = makefig(rotation, hex_side_length, hcoord_translated, vcoord_translated, img, output_path)

    
    # Save parameters to json file
    save_params(derivatives_base, radius, hex_side_length, theta, desired_x, desired_y, rotation, hcoord_translated, vcoord_translated, consinks = True)

    print(f"Maze overlay saved to {output_path}")

    if good_overlay:
        print("Overlay approved, continuing all operations")
    else:
        print("Overlay not approved. Adjust parameters in overlay_maze_image_fromVideo function")
        print('Spatial processing pipeline will assign platforms to positional csvs.\n ')
    return good_overlay, img

if __name__ == "__main__":
    derivatives_base = Path(r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801")
    overlay_maze_image_consinks(derivatives_base, "video")


