
import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.axes._axes import Axes 
from matplotlib.figure import Figure
from tqdm import tqdm


def plot_heatmap_xy(derivatives_base: Path, trials_to_include: list, goals_to_include: list = [0,1,2], frame_rate: int = 25):
    """
    Makes a heatmap of the xy position of the animal for each trial. 
    This is used to check whether all xy positions are correct

    Inputs
    -----
    derivatives_base (Path): Path to derivatives folder
    trials_to_include (list): list of trials numbers
    frame_rate (int: 25): frame rate of camera

    Outputs
    -----
    saved in spatial_behav_data/position_heatmaps

    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives","rawdata")).parent
    
    # Input and output folders
    spatial_behav_folder = input_folder = derivatives_base/'analysis'/'spatial_behav_data'
    input_folder = spatial_behav_folder/'XY_and_HD'
    output_folder = spatial_behav_folder/'position_heatmaps'
    output_folder.mkdir(exist_ok = True)
    
    pos_data_all_csv = input_folder/'XY_HD_alltrials_center.csv'
    pos_data_all = pd.read_csv(pos_data_all_csv)
    
    # Limits
    # Limits and outline
    xmin, xmax, ymin, ymax = get_limits(derivatives_base)
    outline_x, outline_y = get_outline(derivatives_base)
        
    goals_to_include.append(3) # 3 == full trial
    
    print(f"Saving heatmap plots to {output_folder}")
    # Plot per trial
    plot_heatmap_per_trial(rawsession_folder, pos_data_all, xmin, xmax, ymin, ymax, outline_x, outline_y, input_folder, output_folder, trials_to_include, goals_to_include, frame_rate)

    plot_heatmap_alltrials(spatial_behav_folder,  pos_data_all, xmin, xmax, ymin, ymax, outline_x, outline_y,  output_folder, trials_to_include, goals_to_include, frame_rate)

def plot_heatmap_per_trial(rawsession_folder: Path, pos_data_all: pd.DataFrame, xmin: float, xmax: float, ymin: float, ymax: float, outline_x, outline_y, input_folder: Path, output_folder: Path, trials_to_include: list, goals_to_include: list = [0,1,2], frame_rate: int = 25):
    """ Makes a plot of the heatmap per trial, for the full trial and per goal"""
    
    for tr in tqdm(trials_to_include):
        fig, axs = plt.subplots(1, len(goals_to_include), figsize = [len(goals_to_include)*6, 6])
        fig.suptitle(f"Trial {tr}")
        axs = axs.flatten()
        csv_name = f'XY_HD_center_t{tr}.csv'
        pos_data_tr = pd.read_csv(os.path.join(input_folder, csv_name))

        
        for i_g, g in enumerate(goals_to_include):
            ax = axs[i_g]
            if g == 3: # Meaning full trial 
                pos_data = pos_data_tr   
                title = 'Full trial'     
            else:
                path = os.path.join(rawsession_folder, "task_metadata", "restricted_final.csv")

                intervals_df = pd.read_csv(path)

                start_col = 2*g
                end_col = 2*g + 1
                
                # get only start and end col
                intervals_df_restr = intervals_df.iloc[:, start_col:end_col + 1]
                
                # Convert to list of tuples
                intervals = list(zip(intervals_df_restr.iloc[:,0], intervals_df_restr.iloc[:,1]))
                
                # Convert to frame number
                intervals = [(int(start * frame_rate), int(end * frame_rate)) for start, end in intervals]
                intervals = intervals[tr - 1]
                pos_data_all['frame'] = np.arange(1,len(pos_data_all) + 1)
                mask = np.zeros(len(pos_data_all), dtype=bool)
                start = intervals[0]
                end = intervals[1]
                mask |= (pos_data_all['frame'] >start) & (pos_data_all['frame'] < end)
                pos_data = pos_data_all[mask]
                title = f'Goal {g}'
            x = pos_data['x'].values
            y = pos_data['y'].values

            plot_heatmap(title,x, y, xmin, xmax, ymin, ymax, frame_rate, outline_x, outline_y, ax= ax, fig= fig)
        output_file = output_folder /  f'position_heatmap_trial{tr}.png'
        plt.savefig(output_file, dpi=300)
        plt.close(fig)
    

def plot_heatmap_alltrials(spatial_behav_folder: Path, pos_data_all: pd.DataFrame, xmin: float, xmax: float, ymin: float, ymax: float, outline_x, outline_y,  output_folder: Path, trials_to_include: list, goals_to_include: list = [0,1,2], frame_rate: int = 25):
    """ Lastly plot all trials together, with subplot 1 and 2 also split for goal"""
    fig, axs = plt.subplots(1, len(goals_to_include), figsize = [len(goals_to_include)*6, 6])
    axs = axs.flatten()
    fig.suptitle("Heatmap for all trials")
        

    for i_g, g in enumerate(goals_to_include):
        
        ax = axs[i_g]
        if g == 3: # Meaning full trial
            pos_data = pos_data_all
            title = 'All trials'
        else:
            path = spatial_behav_folder / 'XY_and_HD'/ f'XY_HD_goal{g}_trials.csv'
            pos_data = pd.read_csv(path)
            title = f'Goal {g}'

        x = pos_data['x'].values
        y = pos_data['y'].values
        
        plot_heatmap(title,x, y, xmin, xmax, ymin, ymax, frame_rate, outline_x, outline_y, ax= ax, fig= fig)
        
    plt.savefig(os.path.join(output_folder, 'position_heatmap_alltrials.png'), dpi=300)
    plt.close(fig)
    print(f"Saved heatmaps to {output_folder}")
     
def get_limits(derivatives_base: Path) -> tuple[float, float, float, float]:
    """ Reads in limits from limits.json"""
    limits_path = derivatives_base/"analysis"/"maze_overlay"/"limits.json"
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    
    xmin = limits['x_min']
    xmax = limits['x_max']
    ymin = limits['y_min']
    ymax = limits['y_max']
    return xmin, xmax, ymin, ymax

def get_outline(derivatives_base: Path):
    """Obtains outline of maze from maze_outline_coords.json"""
    outline_path = derivatives_base/"analysis"/"maze_overlay"/"maze_outline_coords.json"
    if outline_path.exists():
        with open(outline_path, "r") as f:
            outline = json.load(f)
        outline_x = outline["outline_x"]
        outline_y = outline["outline_y"]
    else:
        print("Maze outline JSON not found; skipping red outline overlay.")
        outline_x, outline_y = None, None     
    return outline_x, outline_y
 
def plot_heatmap(title: str,x: np.ndarray, y: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float, frame_rate: int, outline_x: np.ndarray | None, outline_y: np.ndarray | None, ax: Axes, fig: Figure):
    """ Plots a heatmap"""
    x = x[~pd.isna(x)]
    y = y[~pd.isna(y)]
    heatmap_data, xbins, ybins  =  np.histogram2d(x, y, bins=20, range=[[xmin, xmax], [ymin, ymax]])
    heatmap_data = heatmap_data/frame_rate
    
    im = ax.imshow(
        heatmap_data.T,
        cmap='viridis',
        interpolation=None,
        origin='upper',
        aspect='auto',
        extent=[xmin, xmax, ymax, ymin]
    )
    fig.colorbar(im, ax=ax, label='Seconds')
    if outline_x is not None and outline_y is not None:
        ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    ax.set_title(title)
    ax.set_aspect('equal')
         
if __name__ == "__main__":
    derivatives_base= r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"
    trials_to_include = np.arange(1,10)
    #plot_heatmap_xy(derivatives_base, trials_to_include)
    pos_data_path = os.path.join(derivatives_base, 'analysis', 'spatial_behav_data', 'XY_and_HD', 'XY_HD_goal2_trials.csv')
    pos_data_g2 = pd.read_csv(pos_data_path)
    print(len(pos_data_g2))