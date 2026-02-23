# preprocessing/spikewrap_pipeline.py
'''
Script to run spikewrap

'''
import spikewrap as sw
from pathlib import Path
import torch
from preprocessing.append_config import append_config
print(torch.cuda.is_available()) # Checks whether GPU is in use

def run_spikewrap(derivatives_base: Path, subject_path: Path, session_name: str, concat_runs: bool = True) -> None:
    """
    Function runs spikewrap
    NOTE: if concat_run is false, the folder name is different than usual (usually its concat_run), leading to errors in the spikeinterface code
    
    Inputs
    --------
    derivatives_base: Path
        path to derivatives folder
    subject_path: Path
        path to ephys data
    session_name: str
        name of this session (for example, all_trials)

    Creates
    ---------
        Binned data (as a raw file)
        Kilosort4 output

    Currently runs the following
    ----------------
    Preprocessing:
        Global CAR
        Bandpass filter 300-6000 Hz
    Kilosort4:
        No drift correction (nblocks = 0)
        No CAR
        Highpass cutoff 100 Hz (probably redundant given preprocessing)
    
    Called by
    ---------
    main_pipeline.py
    """
    
    session = sw.Session(
        subject_path=subject_path, # path to rawdata/sub-001_id-XX
        session_name=session_name, # For example, ses-01
        file_format="spikeglx",
        run_names="all",
        output_path = derivatives_base /  "ephys"
    )

    session.preprocess(
        configs="neuropixels+kilosort2_5",
        per_shank=False,
        concat_runs=concat_runs,
    )


    #plots = session.plot_preprocessed(time_range=(0, 0.1), show=True)

    session.save_preprocessed(
        overwrite=True,
        n_jobs=1,
        slurm=False
    )

    # Parameters for running kilosort
    do_CAR = False
    save_preprocessed_copy = True
    nblocks = 0 # Turns off drift correct
    highpass_cutoff = 100
    
    # Saving params to config file
    append_config(derivatives_base, {'spikewrap': {'do_CAR': do_CAR, 'save_preprocessed_copy': save_preprocessed_copy, 'nblocks': nblocks , 'highpass_cutoff': highpass_cutoff}})
    
    cfg = sw.load_config_dict(sw.get_configs_path() / "neuropixels+kilosort2_5.yaml")
    del cfg["sorting"]["kilosort2_5"]
  #  settings = {"n_chan_bin": 384, "nblocks": 0 , "highpass_cutoff": 100}  # nblocks=0 turns off drift correction, I think you can also do it with a "do_correction" (spikeinterface)
    #         "use_binary_file": True,
    #    "delete_recording_dat": True,
    cfg["sorting"]["kilosort4"] = {"do_CAR": False, "save_preprocessed_copy": True, "save_preprocessed_copy": False, "nblocks": 0 , "highpass_cutoff": 100}

    session.sort(cfg, run_sorter_method="local", per_shank=False, concat_runs=False)

if __name__ == "__main__":
    derivatives_base = "D:/Spatiotemporal_task/derivatives/sub-003_id_2V/ses-02_testHCT/test"
    subject_path = "D:/Spatiotemporal_task/rawdata/sub-003_id_2V"
    session_name = 'ses-02_testHCT'
    run_spikewrap(derivatives_base, subject_path, session_name)