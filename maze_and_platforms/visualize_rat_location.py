import json
import glob
import cv2
from find_platforms import get_platform_number, makefig
from pathlib import Path

def visualize_rat_location(derivatives_base: Path,
                           ratloc_x: int, 
                           ratloc_y: int) -> None:
    """
    Overlays a point (ratloc_x, ratloc_y) onto a frame of the video of the trials, returns which platform it is on
    and also overlays the hexagonal grid
    
    Inputs
    ---------
    derivatives_base: Path 
        Path to derivatives folder
    ratloc_x: int 
        x location that you want to overlay
    ratloc_y: int
        y location that you want to overlay
    
    """
    # Loading raw session folder
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    # Loading hexagon parameters
    params_path = derivatives_base / "analysis" / "maze_overlay" / "maze_overlay_params.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    # Getting image
    tracking_dir = rawsession_folder / "tracking"
    files = list(tracking_dir.glob("T*.avi"))
    video_path = files[0]
    cap = cv2.VideoCapture(video_path)

    ret, img = cap.read()   # img will hold the first frame as a NumPy array
    cap.release()
    if not ret:
        print("Failed to read first frame")

    cap.release()
    # Getting platform numbe
    plat = get_platform_number(ratloc_x, ratloc_y,params['hcoord_tr'], params['vcoord_tr'], params['hex_side_length'] )
    print(f"Platform is {plat}")
    makefig(params['rotation'],  params['hex_side_length'], params['hcoord_tr'], params['vcoord_tr'], img, ratloc_x = ratloc_x, ratloc_y = ratloc_y)
    
if __name__ == "__main__":
    x = 1308
    y = 1276
    
    derivatives_base = Path(r"C:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials")

    visualize_rat_location(derivatives_base, x, y)