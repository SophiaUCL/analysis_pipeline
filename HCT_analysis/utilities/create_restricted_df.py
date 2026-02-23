from HCT_analysis.utilities.create_intervals_specialbehav import create_intervals_specialbehav, check_restricted_df_exists, make_restricted_df
from HCT_analysis.utilities.trials_utils import append_alltrials
from pathlib import Path
from HCT_analysis.utilities.concat_trials import concat_trials

def create_restricted_df(derivatives_base: Path, goals_to_include: list, trials_to_include: list):
    """ Runs append all trials, which takes the alltrials csv and creates a new csv only with the rows of the trial date
    If restricted_df doesn't exist, it creates the restricted df and creates the intervals for special behavior. If it does exist, it only creates the intervals for special behavior (which are needed for the HCT pipeline)
    NOTE: better to create it manually than automatically (That's just based on the labview data, not on the behavioural data)
    
    This function is mainly created to make spatial_processing_pipeline a bit cleaner. 
    
    Inputs
    -----
    derivatives_base (Path): Path to derivatives folder
    goals_to_include (list): list with goal numbers that were run for these recordings
    trials_to_include (list): list with trial numbers that were run for these recordings
    
    Outputs
    -----
    derivatives_base/rawsession/behaviour/restricted_df.csv: csv with only the rows of the trial date and only the goals that are included. This is used for the HCT pipeline to create the intervals for special behavior. 
    rawsession/behaviour/alltrials_trialday.csv alltrials filtered for only this day
    
    Called by
    --------
    spatial_processing_pipeline.py
    """
    # Creates trial csv only for thsi day
    append_alltrials(derivatives_base)

    # Checks whether restricted df exists
    restricted_df_exists =  check_restricted_df_exists(derivatives_base)
    if restricted_df_exists:
        print("Restricted_df exists")

    # If it doesn't exist and 0 is not in goals to include, it creates the restricted df and creates the intervals for special behavior. If it does exist, it only creates the intervals for special behavior (which are needed for the HCT pipeline)
    if not restricted_df_exists and 0 not in goals_to_include:
        make_restricted_df(derivatives_base, goals_to_include, trials_to_include)
        create_intervals_specialbehav(derivatives_base)
    elif restricted_df_exists:
        create_intervals_specialbehav(derivatives_base)
    else:
        print("0 in goals to include and restricted_df.csv doesn't exist. Please create manually and rerun")
        
    concat_trials(derivatives_base)