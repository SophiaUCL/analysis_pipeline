import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json
#from utilities.add_platforms_any_df import add_platforms_to_csv

def restrict_posdata_specialbehav(derivatives_base: Path, goals_to_include: list = [0,1,2], show_plots: bool = True, frame_rate: int = 25):
    """
    Restricts the pos_data to the intervals of the goal.
    
    Inputs
    ------
    derivatives_base (Path): Path to derivatives folder
    goals_to_include (list: [0,1,2]): Goals included in analysis
    show_plots (bool: False): Whether to show the plots
    frame_rate (int: 25)

    Returns:
        DataFrame: restricted position data
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent

    pos_data_path = derivatives_base/'analysis'/'spatial_behav_data'/'XY_and_HD'/'XY_HD_w_platforms.csv'
    pos_data = pd.read_csv(pos_data_path)
    
    # Limits
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    
    xmin = limits['x_min']
    xmax = limits['x_max']
    ymin = limits['y_min']
    ymax = limits['y_max']
    
    # ---- Load maze outline coordinates ----
    outline_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline_coords.json")
    if os.path.exists(outline_path):
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        print(" Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None
        
    for goal in goals_to_include:
        pos_data_org = pos_data.copy()
        
        
        path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")

        intervals_df = pd.read_csv(path)

        start_col = 2*goal
        end_col = 2*goal + 1
        
        # get only start and end col
        intervals_df_restr = intervals_df.iloc[:, start_col:end_col + 1]
        
        # Convert to list of tuples
        intervals = list(zip(intervals_df_restr.iloc[:,0], intervals_df_restr.iloc[:,1]))
        
        # Convert to frame number
        intervals = [(int(start * frame_rate), int(end * frame_rate)) for start, end in intervals]

        
        pos_data_org['frame'] = np.arange(1,len(pos_data_org) + 1)
        mask = np.zeros(len(pos_data), dtype=bool)
        for start, end in intervals:
            mask |= (pos_data_org['frame'] >start) & (pos_data_org['frame'] < end)
        pos_data_org = pos_data_org[mask]
        
        print(f"Len dataframe for goal {goal}: {len(pos_data_org)}")
        output_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{goal}_trials.csv')
        pos_data_org.to_csv(output_path, index=False)
        print(f"Saved restricted position data for goal {goal} to {output_path}")

    # Saving all into one df
    df_restricted_all =  pd.DataFrame(columns=['x', 'y', 'hd'])
    for goal in goals_to_include:
        df_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', f'XY_HD_goal{goal}_trials.csv')
        df_goal = pd.read_csv(df_path)
        df_restricted_all = pd.concat([df_restricted_all, df_goal])
        df_restricted_all = df_restricted_all.sort_values(by='frame').reset_index(drop=True)

    print("Adding platforms to interval df")
    output_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_allintervals.csv')
    df_restricted_all.to_csv(output_path, index = False)
    
    goals_to_plot = np.append(goals_to_include, [3,4]) if len(goals_to_include) > 1 else np.append(goals_to_include, 3)
    fig, ax = plt.subplots(1, len(goals_to_plot), figsize=(len(goals_to_plot)*4, 4))
    ax = ax.flatten()
    
    for i, goal in enumerate(goals_to_plot):
        if goal < 3:
            goal_path = os.path.join(
                derivatives_base,
                'analysis',
                'spatial_behav_data',
                'XY_and_HD',
                f'XY_HD_goal{goal}_trials.csv'
            )
            goal_df = pd.read_csv(goal_path)
            title = f'Goal {goal}'
        elif goal == 3:
            goal_df = pos_data
            title = 'All trials'
        else:
            goal_df = df_restricted_all
            title = 'All trials, only during intervals'
        
        x = goal_df['x'].values
        if not x.size: # empty array
            continue
        x = x[~np.isnan(x)]
        y = goal_df['y'].values
        y = y[~np.isnan(y)]


        heatmap_data, xbins, ybins  = np.histogram2d(x, y, bins=30, range=[[xmin, xmax], [ymin, ymax]])
        
        im = ax[i].imshow(
                heatmap_data.T,
                cmap='viridis',
                interpolation=None,
                origin='upper',
                aspect='auto',
                extent=[xmin, xmax, ymax, ymin]
            )
        fig.colorbar(im, ax=ax[i], label='Seconds')
        ax[i].set_title(title)
        ax[i].set_aspect('equal')
        if outline_x is not None and outline_y is not None:
            ax[i].plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    plt.tight_layout()
    output_folder = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data',"occupancy_heatmaps")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, 'session_overview.png')
    plt.savefig(output_path)
    if show_plots:
        plt.show()
    plt.close(fig)
                



    
if __name__ == "__main__":
    derivatives_base= r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"


    restrict_posdata_specialbehav(derivatives_base)
    
        