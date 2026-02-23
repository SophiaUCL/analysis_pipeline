import numpy as np
from pathlib import Path
from unit_features.plot_firing_each_epoch import plot_firing_each_epoch

""" This file contains the code to calculate the popsink and consink. 
Run this pipeline after the spatial processing pipeline

Ensure you have classified your cells before this
"""
trials_to_include= np.arange(1,27)
print(trials_to_include)
derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
derivatives_base = Path(derivatives_base)



plot_firing_each_epoch(derivatives_base, trials_to_include, unit_type = "all")
degrees_df_path, deg = make_spatiotemp_plots(derivatives_base,trials_to_include, unit_type = "all", make_plots = False)
make_roseplots(derivatives_base, trials_to_include, deg, path_to_df = degrees_df_path)
sig_across_epochs(derivatives_base, trials_to_include)
    