import subprocess
import pathlib
import os
import numpy as np
import json
import sys
import shutil

print(">>> Script started")
sys.stdout.flush()

"""
SAME AS OTHER SCRIPT, BUT THIS ONE YOU CAN DIRECTLY RUN IN THE SLEAP ENVIRONMENT
this is a quick script that allows you to run your Sleap inference on video files in a directory by calling subprocess. 
It will create a new folder in the OUTPUT_FOLDER with the same structure as the ROOT_FOLDER and save the inference results there.
it will save a .slp file but also an .h5 file for further analysis for example with movement (https://movement.neuroinformatics.dev/index.html)


-------
Parameters:
video_folder: the folder where your video files are stored
dest_folder: the folder where you want to save the inference results
centroid_model_folder: folder where centroid model is
centered_model_folder: folder where centered model is


call_inference()
params:
fpath: the path to the video file you want to run inference on
dest_folder: the folder where you want to save the inference results
command_inf: the command to run the inference, you need to change the model paths to your own models
and potentially adjust inference parameters.


when you want to run this script you need to have sleap installed in the environment where you run this script. 
then just open a command line terminal activate the environment and type: python run_inference_on_all.py
--------
"""

def call_inference_on_all(derivatives_base, centroid_model_folder, centered_model_folder,  all_trials = True, ext=".avi"):
    """
    Runs centroid model and centered model on the videos found in the rawdata/tracking folder
    Saves inference results in derivatives_base\analysis\spatial_behav_data\inference_results
    """
    # Loading folders
    rawsession_folder = derivatives_base.replace("derivatives", "rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    video_folder = os.path.join(rawsession_folder, 'tracking')
    dest_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'inference_results')
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    source_folder = pathlib.Path(video_folder)
    
    # Loading videopaths
    fpaths = list(source_folder.rglob(f"*{ext}"))
    counter = 0
    for fpath in fpaths:
        if not all_trials and counter > 0: # if you just want to do a single trial, then we stop here
            break 
        relative_path_parent = fpath.relative_to(source_folder).parent
        dest_path = pathlib.Path(dest_folder) / relative_path_parent
        call_inference(fpath, dest_path, centered_model_folder, centroid_model_folder)
        counter += 1
    return fpaths


def call_inference(fpath, dest_folder, centered_model_folder, centroid_model_folder):
    fpath = pathlib.Path(fpath)
    dest_path = dest_folder / f"{fpath.stem}_inference.slp"

    if dest_path.exists():
        # Skips video if inference file already exists
        print(f"Skipping {fpath} as {dest_path} already exists.\n")
        return

    if fpath.exists():
        # Runs preprocessing
        print(f"processing: {fpath}\n")
        command_inf = f"sleap-track -m {centered_model_folder} -m {centroid_model_folder} -o {dest_path}  --tracking.tracker none {fpath}  "
        print(f"running inference: {command_inf}\n")
        subprocess.call(command_inf, shell=True)
        final_dest_path = dest_folder / f"{fpath.stem}.h5"
        command_conv = f"sleap-convert --format analysis -o {final_dest_path} {dest_path} "
        print("Which sleap-track is being used:", shutil.which("sleap-track"))
        print(f"converting to h5: {final_dest_path}\n")
        subprocess.call(command_conv, shell=True)

    else:
        raise FileNotFoundError(f"File {fpath} does not exist. Please check your input.")




if __name__ == "__main__":
    # Currently running in sleap_new environment from Sopia's computer, and sleap environment on Eylon's
    #derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-003_id-2F\ses-02_date-18092025\all_trials"
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
    centroid_model_folder = r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\SLEAP_NEWCAMERA_21072025\models\latest_model\251003_111713.centroid.n=2405"
    centered_model_folder = r"\\ceph-gw02.hpc.swc.ucl.ac.uk\okeefe\Eylon\SLEAP_NEWCAMERA_21072025\models\251003_132856.centered_instance.n=2405"
    all_trials = True  # If you just want to test run it, then set this to False
    call_inference_on_all(derivatives_base, centroid_model_folder, centered_model_folder, all_trials = all_trials)
