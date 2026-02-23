from pathlib import Path
from typing import Literal
from maze_and_platforms.find_platforms import get_image, get_params, get_vertices, get_outline, save_outline, plot_outline

OverlayMethod = Literal["video", "image"]
    
def plot_maze_outline(derivatives_base: Path, 
                      img = None, 
                      method: OverlayMethod = "video"):
    """
    Plots and saves the outer boundary of the honeycomb maze 
    (edges of the outermost platforms), and saves outline coordinates to JSON.
    
    Inputs
    ---------
    derivatives_base: Path
        Path to derivatives base directory (e.g. .../derivatives/sub-XXX/.../all_trials)
    img: optional
        image to overlay on
    method: "video" or "image"
        Method of overlay (used if image is none)
    
    Outputs
    --------
        derivatives_base\analysis\maze_overlay\maze_outline_coords.json
            outline coordinates for maze
        derivatives_base\analysis\maze_overlay\maze_outline.png
            image with maze outline visualised
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    # Loading data
    hcoord, vcoord, side, rotation_deg = get_params(derivatives_base)

    if img is None:
        img = get_image(rawsession_folder, method = method)
        
    # Location of all vertices
    all_vertices = get_vertices(hcoord, vcoord, rotation_deg, side)

    # get outline
    outline_x, outline_y = get_outline(all_vertices)
    
    # saving
    save_outline(derivatives_base, outline_x, outline_y)

    # plotting 
    plot_outline(derivatives_base, img, outline_x, outline_y, hcoord, vcoord)

if __name__ == "__main__":
    derivatives_base = Path(r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801")
    plot_maze_outline(derivatives_base)
