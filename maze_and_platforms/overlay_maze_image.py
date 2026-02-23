import numpy as np
from pathlib import Path
from typing import Literal

from maze_and_platforms.find_platforms import (
    hex_grid,
    calculate_cartesian_coords,
    translate_coords,
    save_params,
    get_image,
    makefig,
)

PlatformWidth = 20.2 # Width of platform in cm
OverlayMethod = Literal["video", "image"]

def overlay_maze_image(derivatives_base: Path, 
                       method: OverlayMethod,
                       )-> tuple[bool, np.ndarray]:
    """
    Overlay a hexagonal maze grid onto a session image.

    Depending on `method`, the overlay is applied either to:
    - the reference maze image ("image"), or
    - the first frame of the session video ("video").

    The resulting image and overlay parameters are saved under:
    derivatives/analysis/maze_overlay/
    
    NOTE: use overlay_hexagon_ipynb.ipynb to optimize the overlay parameters interactively (its a bit easier than rerunning this function)
    Also note that the overlay won't be perfect for all squares due to the angle of the camera, etc. Best to optimize the overlay so it works well for the center platform

    Parameters
    ----------
    derivatives_base : Path
        Path to the session's derivatives directory.
    method : {"video", "image"}
        Source used for the overlay.

    Returns
    -------
    good_overlay: bool
        Whether the overlay was visually approved.
    img : np.ndarray
        Image used for generating the overlay.
    """
    # Loading rawsession folder
    rawsession_folder = Path(
        str(derivatives_base).replace("derivatives", "rawdata")
    ).parent
    
    # Obtaining image used for overlay
    img = get_image(rawsession_folder, method)

    # Output folder
    output_path = derivatives_base / "analysis" / "maze_overlay" / "maze_overlay.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parameters that control the overlay
    radius = 4
    hex_side_length = 88 # Length of the side of the hexagon
    theta = np.radians(305)  # Rotation angle in radians
    desired_x, desired_y = 1340, 970 # center of 31st platform
    coord = hex_grid(radius) # coordinates
    rotation = 25

    
    # Calculate initial Cartesian coordinates
    hcoord2, vcoord2 = calculate_cartesian_coords(coord, hex_side_length)
    hcoord_translated, vcoord_translated = translate_coords(hcoord2, vcoord2, theta, desired_x, desired_y, num_plats = 61)
    
    # plots figure
    good_overlay = makefig(rotation,hex_side_length, hcoord_translated, vcoord_translated, img, output_path)
    
    # Calculate the number of pixels per cm
    pixels_per_cm = calculate_pixels_per_cm(hcoord_translated, vcoord_translated)
    
    # Save parameters to json file
    save_params(derivatives_base, radius, hex_side_length, theta, desired_x, desired_y, rotation, hcoord_translated, vcoord_translated, pixels_per_cm)

    print(f"Maze overlay saved to {output_path}")

    
    if good_overlay:
        print("Overlay approved, continuing all operations")
    else:
        print("Overlay not approved. Adjust parameters in overlay_maze_image_fromVideo function")
        print('Spatial processing pipeline will assign platforms to positional csvs.\n ')
    return good_overlay, img


def calculate_pixels_per_cm(hcoord, vcoord):
    """ Calculates the number of pixels per cm based on the distance between the center platforms"""

    distances = []
    
    # Calculate the distance between the first 4 platforms sequentially
    for i in range(4):
        dx = hcoord[i+1] - hcoord[i]
        dy = vcoord[i+1] - vcoord[i]
        distance = np.sqrt(dx**2 + dy**2)
        distances.append(distance)
        
    # Calculate average pixels per cm
    avg_distance_pixels = np.mean(distances)
    pixels_per_cm = avg_distance_pixels / PlatformWidth
    print(f"Pixels per cm: {pixels_per_cm}")
    return pixels_per_cm
    



if __name__ == "__main__":
    derivatives_base =  Path(r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801")
    overlay_maze_image(derivatives_base, "video")
