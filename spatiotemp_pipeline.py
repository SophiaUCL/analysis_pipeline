import numpy as np
from pathlib import Path
from spatiotemporal_analysis.plot_firing_each_epoch import plot_firing_each_epoch
from spatiotemporal_analysis.roseplot import make_roseplots
from spatiotemporal_analysis.make_spatiotemp_plots import make_spatiotemp_plots
from spatiotemporal_analysis.HD_across_epoch import sig_across_epochs
from spatiotemporal_analysis.get_MRL_significance import get_MRL_significance
""" This file contains the code to calculate the popsink and consink. 
Run this pipeline after the spatial processing pipeline

Ensure you have classified your cells before this
"""
trials_to_include= np.arange(1,27)
print(trials_to_include)
derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
derivatives_base = Path(derivatives_base)
unit_type = "pyramidal"


# spike count for each trial each for each unit
plot_firing_each_epoch(derivatives_base, trials_to_include, unit_type = unit_type)

# makes spatiotemproal plots
degrees_df_path, deg = make_spatiotemp_plots(derivatives_base,trials_to_include, unit_type = unit_type, make_plots = False)

# If you only want significance and not the plots, uncomment the one below
#degrees_df_path, deg = get_MRL_significance(derivatives_base, trials_to_include)

# Makes roseplots
make_roseplots(derivatives_base, trials_to_include, deg, path_to_df = degrees_df_path)

# PLots signifiance across units
sig_across_epochs(derivatives_base, trials_to_include)
    