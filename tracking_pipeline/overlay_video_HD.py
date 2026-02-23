import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path



def overlay_video_HD(derivatives_base: Path, trials_to_include: list, short: bool = False):
    """
    Overlays videos with the position of the animal and the head direction

    Inputs
    -------
    derivatives_base (Path): path to derivatives folder
    trials_to_include (list): trials to include in this analysis
    short (bool: False): Whether to only overlay the first 30 seconds
    
    Outputs
    -------
    derivatives_base\analysis\processed_video\T{tr}_with_HD.avi: video file with overlayed position and HD
    
    Notes
    -----
    Videos must have the format T{trial number}_*.avi (where * denotes that anything can be there)
    HD is assumed to be in radians in the dataframes
    
    """
    # Getting idrectories
    output_dir, video_data_dir, pos_data_dir = get_dirs(derivatives_base)
   
    # Go over all trials
    for tr in trials_to_include:
        # Path for trial csv posdata 
        trial_csv_path = pos_data_dir/f'XY_HD_t{tr}.csv'
        df = pd.read_csv(trial_csv_path)
        
        # Path for output
        output_path = output_dir/f"t{tr}_with_HD.avi"
        print(f"Output path for video overlay: {output_path}")

        # Finding video path
        files = list(video_data_dir.glob(f"*T{tr}_*.avi"))

        video_path = files[0]
        do_overlay(video_path, df, output_path, short = short)


def do_overlay(video_path: Path, df: pd.DataFrame, output_path: Path, short: bool):
    """ 
    Function overlays the video with the head direction and xy position

    Inputs: 
    video_path: path to original video
    df: path to df with x, y, and hd data (in that order)
    output_path: path where video is saved

    Note:
    Its assumed that df has hd in radians
    """
    # Positional data
    head_pos_x_col = df.iloc[:, 0].to_numpy()
    head_pos_y_col = df.iloc[:, 1].to_numpy()
    hd_col = df.iloc[:, 2].to_numpy()
    
    if np.nanmax(hd_col) > 2*np.pi:
        raise ValueError("hd is in degrees. Rerun movement to get data in radians")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total frame count

    pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if short and frame_idx > 25*30:
            break
        if frame_idx < len(head_pos_x_col) and pd.notna(head_pos_x_col[frame_idx]) and pd.notna(head_pos_y_col[frame_idx]) and pd.notna(hd_col[frame_idx]):
            x = int(head_pos_x_col[frame_idx])
            y = int(head_pos_y_col[frame_idx])
            hd = hd_col[frame_idx]

            # Draw a red square of 5x5 pixels at the head_pos
            size = 5
            cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), (0, 0, 255), -1)

            # Calculate the end point of the arrow
            length = 30  # Length of the arrow
            end_x = int(x + length * np.cos(hd))
            end_y = int(y - length * np.sin(hd))

            # Draw the arrow
            cv2.arrowedLine(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)
    pbar.close()

    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

def get_dirs(derivatives_base: Path):
    """ Getting all the directories needed for function"""
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    # Getting data paths
    pos_data_dir = derivatives_base/'analysis'/'spatial_behav_data'/'XY_and_HD'
    if not pos_data_dir.exists():
        raise FileNotFoundError(f"Positional data directory does not exist: {pos_data_dir}")

    video_data_dir = rawsession_folder/"tracking"
    if not video_data_dir.exists():
        raise FileNotFoundError(f"Video data directory does not exist: {video_data_dir}")
    
    output_dir = derivatives_base/"analysis"/"processed_video"
    output_dir.mkdir(exist_ok=True)
    
    return output_dir, video_data_dir, pos_data_dir


if __name__ == "__main__":
    derivatives_base = Path(r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801")
    overlay_video_HD(derivatives_base, [25], False)
        