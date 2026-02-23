import numpy as np
from pathlib import Path
from HCT_analysis.find_consinks_main import main as find_consinks_main
from HCT_analysis.get_directional_occupancy_by_pos import main as directional_occupancy_main
from HCT_analysis.find_popsink_main import calculate_popsink
from spatial_features.combine_autowv_ratemaps import combine_autowv_ratemaps_vectorfields
from HCT_analysis.calculate_vectorfields_newmethod import main as calculate_vectorfields

""" This file contains the code to calculate the popsink and consink. 
Run this pipeline after the spatial processing pipeline

Ensure you have classified your cells before this
"""
trials_to_include= np.arange(1,27)
print(trials_to_include)
derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
derivatives_base = Path(derivatives_base)
goals_to_include = [1] 

directional_occupancy_main(derivatives_base, goals_to_include = goals_to_include)

# Find consinks
find_consinks_main(derivatives_base, rel_dir_occ='intervals',unit_type= "pyramidal",  goals_to_include = goals_to_include, show_plots = True, code_to_run=[3,4])


calculate_vectorfields(derivatives_base, unit_type = "pyramidal", goals_to_include = goals_to_include)

# Find popsink
calculate_popsink(derivatives_base, unit_type = "all", load_units_which_method = 2, code_to_run = [1,2], title='Population Sinks', dir = 'popsinks', goals_to_include = [1])
    
combine_autowv_ratemaps_vectorfields(derivatives_base, unit_type = "pyramidal")
    