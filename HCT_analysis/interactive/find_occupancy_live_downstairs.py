
from plot_occ_downstairs import make_plot
import os
import pandas as pd
import glob

""" 
This script is used downstairs to plot the start platforms and occupancy for G1, G2 (if applicable) and full trials

It finds the latest alltrials_{animalID file}
extracts date and animal id
finds all the single trial files for that date and animal id
concatenates those
inputs that into a function which plots the starts and occupancy

"""
# Setting current directory to HCT_analysis
cwd = os.getcwd()
HCT_dir = os.path.dirname(cwd)
os.chdir(HCT_dir)

directory_path = r"C:\Users\JOK\Documents\Honeycomb_task_25\Data\Trial_data"
most_recent_file = None
most_recent_time = 0

# iterate over the files in the directory using os.scandir
for entry in os.scandir(directory_path):
    if entry.is_file():
        # get the modification time of the file using entry.stat().st_mtime_ns
        mod_time = entry.stat().st_mtime_ns
        if mod_time > most_recent_time:
            # update the most recent file and its modification time
            most_recent_file = entry.name
            most_recent_time = mod_time

# we're opening the latest file, which should be of the form "alltrials_{animalid}.csv"
file_path = os.path.join(directory_path, most_recent_file)
df = pd.read_csv(file_path)

# Extracting animal ID
file_name = os.path.basename(file_path)
if "alltrials" not in file_name:
    raise ValueError("File that was found is not of the form alltrials_{animalid}.csv")
animal_id = file_name[10:12]  

# Getting current date
curr_date = df["Date"].iloc[-1]
curr_date = str(curr_date)

# If date is missing first element, we add it
if len(curr_date) < 8:
    curr_date = "0" + curr_date

pattern = os.path.join(directory_path, f"{animal_id}_{curr_date}*.csv")
global_matches = glob.glob(pattern)

# Add each trial csv to df_list
df_list = []
for file in global_matches:
    df = pd.read_csv(file)
    trial_number = os.path.basename(file).split('_g')[-1].split('.csv')[0]  # Extract trial number from filename
    df['trial_number'] = int(trial_number) # Add trial number as a new column
    df_list.append(df)

concatenated_df = pd.concat(df_list, ignore_index=True)

goals = concatenated_df["goal"].to_list()
goal_platforms = [x for i, x in enumerate(goals) if goals.index(x) == i] 
if len(goal_platforms) == 1:
    goals_to_include = [1]
else:
    goals_to_include = [1,2, 3]

day = curr_date[0:2]
month = curr_date[2:4]
year = curr_date[4:9]

date = day + "/" + month + "/" + year
make_plot(concatenated_df, goal_platforms, goals_to_include, title = f"Occupancy and starts for {animal_id} on {date}")


