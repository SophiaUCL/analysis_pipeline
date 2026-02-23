import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib
from pathlib import Path
matplotlib.use('TkAgg')
from HCT_analysis.utilities.trials_utils import  get_goal_numbers

def make_maze_behaviour_plots(derivatives_base: Path, goals_to_include: list = [0,1,2], show_plots: bool = True):
    """ 
    Runs all three plotting functions: plot_occupancy, plot_propcorrect, and plot_startplatforms
    
    Inputs
    ------
    derivatives_base (Path): path to derivatives folder
    goal_platforms (list): list of goal platforms [goal1, goal2]
    goals_to_include (list): list of goals that we want to include
    show_plots (bool: True): whether to show the plots
    
    Exports
    -------
    Into {derivatives_base}/analysis/maze_behaviour:
        occupancy.png
        proportion_correct.png
        start_platforms.png
    """
    goal_platforms = get_goal_numbers(derivatives_base)
    
    # Shows a maze plot with the occupancy of each platform for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_occupancy(derivatives_base, goal_platforms, goals_to_include=goals_to_include, show_plots= show_plots)

    # Shows a maze plot with all the start platforms
    plot_startplatforms(derivatives_base, goal_platforms, show_plots= show_plots, goals_to_include=goals_to_include)

    # Shows a maze plot with the proportion of correct choices for goal 1 (left) goal 2 (middle) and all trials (right)
    plot_propcorrect(derivatives_base,goal_platforms, goals_to_include=goals_to_include, show_plots= show_plots)


def plot_startplatforms(derivatives_base: Path, goal_platforms: list, goals_to_include: list = [0,1,2], show_plots: bool = True):
    """
    Shows a maze plot with all the start platforms
    
    Inputs
    ------
    derivatives_base (Path): path to derivatives folder
    goal_platforms (list): list of goal platforms [goal1, goal2]
    goals_to_include (list): list of goals that we want to include
    show_plots (bool: True): whether to show the plots
    

    Exports
    --------
    start_platforms.png into {derivatives_base}/analysis/maze_behaviour
    
    Called by
    -------
    spatial_processing_pipeline.py
    """
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    path = rawsession_folder/"behaviour"/"concatenated_trials.csv"
    df = pd.read_csv(path)
    
    trial_numbers = df['trial_number'].unique()
    
    start_platforms = []
    for t in trial_numbers:
        df_trial = df[df['trial_number'] == t]
        start_platform = df_trial['start platform'].iloc[0]
        start_platforms.append(start_platform)
        
    fig, ax = plt.subplots(1, figsize=(6,6))
    ax.set_aspect('equal')

    hcoord, vcoord, hcoord_rotated, vcoord_rotated = get_coords()
    # Add some coloured hexagons and adjust the orientation to match the rotated grid
    for i, (x, y) in enumerate(zip(hcoord_rotated, vcoord_rotated)):
        if i + 1 in start_platforms:
            colour = "red"
        elif i + 1 == goal_platforms[0]:
            colour = "green"
        elif i + 1 == goal_platforms[1] and len(goals_to_include) > 1:
            colour = "blue"
        else:
            colour = "grey"
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                            orientation=np.radians(60),  # Rotate hexagons to align with grid
                            facecolor=colour, alpha=0.2, edgecolor='k')
        ax.text(x, y, i + 1,  ha='center', va='center', size=15)  # Start numbering from 1
        ax.add_patch(hex)
    start_patch = mpatches.Patch(color="red", alpha=0.2, label="Starts")
    goal_patch = mpatches.Patch(color="green", alpha=0.2, label="Goal 1")
    if len(goals_to_include) > 1:
        goal2_patch = mpatches.Patch(color="blue", alpha=0.2, label="Goal 2")
        ax.legend(handles=[start_patch, goal_patch, goal2_patch], loc="upper right")
    else:
        ax.legend(handles=[start_patch, goal_patch], loc="upper right")
        # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
    ax.set_title('Start platforms (red) and goal platforms (green)')
    output_folder = derivatives_base/"analysis"/"maze_behaviour"
    output_folder.mkdir(exist_ok = True)
    output_file= output_folder / "start_platforms.png"
    plt.savefig(output_file, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)

    print(f"File saved to output folder: {output_folder}")
    

def plot_occupancy(derivatives_base: Path, goal_platforms: list, goals_to_include: list = [0,1,2], show_plots: bool=True) -> None:
    """
    Shows a maze plot with the occupancy of each platform for goal 1 (left) goal 2 (middle) (if applicable) and all trials (right)
    low: red, high: green

    Inputs
    ------
    derivatives_base (Path): path to derivatives folder
    goal_platforms (list): list of goal platforms [goal1, goal2]
    goals_to_include (list): list of goals that we want to include
    show_plots (bool: True): whether to show the plots
    
    Exports
    -------
        occupancy.png into {derivatives_base}/analysis/maze_behaviour
        
    Called by
    -------
    spatial_processing_pipeline.py
    """
    goals_to_include = [el for el in goals_to_include if el != 0]
    if len(goals_to_include) > 1:
        goals_to_include = np.append(goals_to_include, 3)
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    # Input path
    path = rawsession_folder/"behaviour"/"concatenated_trials.csv"
    df = pd.read_csv(path)

    # 3 x 1 subplot
    fig, ax = plt.subplots(1,len(goals_to_include), figsize=(6*len(goals_to_include),6))
    ax = np.atleast_2d(ax)
    ax = ax.flatten()
    fig.suptitle("Frequency of visit per platform")
    hcoord, vcoord, hcoord_rotated, vcoord_rotated = get_coords()


    cmap = plt.cm.RdYlGn
    # Go through goals and all trials
    for j, g in enumerate(goals_to_include): # g = 1: goal 1, g = 2: goal 2, g = 3: all trials
        if g == 1:
            df_g = df[df['goal'] == goal_platforms[0]]
            title = "Goal 1"
        elif g == 2:
            df_g = df[df['goal'] == goal_platforms[1]]
            title = "Goal 2"
        else:
            df_g = df
            title = "Full trials"
        occupancy = []
        
        for plat in np.arange(1, 62):
            df_p = df_g[df_g['start platform'] == plat]
            occupancy.append(len(df_p))

        occupancy_norm = occupancy/np.max(occupancy)
        
        # Add some coloured hexagons and adjust the orientation to match the rotated grid
        for i, (x, y) in enumerate(zip(hcoord_rotated, vcoord_rotated)):
            if occupancy[i] > 0:
                colour = cmap(occupancy_norm[i])  # Map normalized occupancy to colormap
            else:
                colour = "grey"
            if (g > 0 and i + 1 == goal_platforms[g-1]) or (g == 0 and i + 1 in goal_platforms):
                    edgecolor = "blue"
                    linewidth = 5
            else:
                edgecolor = "k"
                linewidth = 1
            text = occupancy[i] if occupancy[i] > 0 else " "
            hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                orientation=np.radians(60),  # Rotate hexagons to align with grid
                                facecolor=colour, alpha=0.2, edgecolor=edgecolor, linewidth= linewidth)
            ax[j].text(x, y, text,  ha='center', va='center', size=15)  # Start numbering from 1
            ax[j].add_patch(hex)
            # Also add scatter points in hexagon centres
        ax[j].scatter(hcoord, vcoord, alpha=0, c = 'grey')
        ax[j].set_title(title)
        ax[j].set_aspect('equal')
    output_folder = derivatives_base/"analysis"/"maze_behaviour"
    output_folder.mkdir(exist_ok = True)
    output_file= output_folder / "occupancy.png"

    plt.savefig(output_file, dpi=300)

    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"File saved to output folder: {output_folder}")

def plot_propcorrect(derivatives_base: Path, goal_platforms: list, goals_to_include: list = [0,1,2], show_plots: bool = True):
    """
    Shows a maze plot with the proportion of correct choices for goal 1 (left) goal 2 (middle) (fif applicable) and all trials (right)

    Inputs
    ------
    derivatives_base (Path): path to derivatives folder
    goal_platforms (list): list of goal platforms [goal1, goal2]
    goals_to_include (list): list of goals that we want to include
    show_plots (bool: True): whether to show the plots
    
    Exports
    ------
        proportion_correct.png into {derivatives_base}/analysis/maze_behaviour
    
    Called by
    -------
    spatial_processing_pipeline.py
    """
    
    goals_to_include = [el for el in goals_to_include if el != 0]
    if len(goals_to_include) > 1:
        goals_to_include = np.append(goals_to_include, 3)
    rawsession_folder = Path(str(derivatives_base).replace("derivatives", "rawdata")).parent
    
    path = rawsession_folder/"behaviour"/"concatenated_trials.csv"
    df = pd.read_csv(path)

    # 3 x 1 subplot
    fig, ax = plt.subplots(1, len(goals_to_include), figsize=(6*len(goals_to_include),6))
    fig.suptitle("Proportion correct per platform and for each trial section")
    ax = np.atleast_2d(ax)
    ax = ax.flatten()

    # coordinates
    hcoord, vcoord, hcoord_rotated, vcoord_rotated = get_coords()
    
    # Colour map
    cmap = matplotlib.colormaps['RdYlGn']
    
    # Go through goals and all trials
    for j, g in enumerate(goals_to_include): # g = 1: goal 1, g = 2: goal 2, g = 3: all trials
        if g == 1:
            df_g = df[df['goal'] == goal_platforms[0]]
            title = "Goal 1"
        elif g == 2:
            df_g = df[df['goal'] == goal_platforms[1]]
            title = "Goal 2"
        else:
            df_g = df
            title = "Full trials"
                
        prop_correct_arr = []
        
        for plat in np.arange(1, 62):
            df_p = df_g[df_g['start platform'] == plat]
            num_rows = len(df_p)
            num_correct = len(df_p[df_p['correct choice'] == 1])
            if num_rows > 0:
                prop_correct = num_correct / num_rows
                prop_correct_arr.append(prop_correct)
            else:
                prop_correct_arr.append(np.nan)

        # Add some coloured hexagons and adjust the orientation to match the rotated grid
        for i, (x, y) in enumerate(zip(hcoord_rotated, vcoord_rotated)):
            if np.isnan(prop_correct_arr[i]):
                colour = 'grey'
                text = ""
            else:
                colour = cmap(prop_correct_arr[i])  # Map normalized occupancy to colormap
                text = np.round(prop_correct_arr[i], 1)
            if (g > 0 and i + 1 == goal_platforms[g-1]) or (g == 0 and i + 1 in goal_platforms):
                edgecolor = "blue"
                linewidth = 5
            else:
                edgecolor = "k"
                linewidth = 1
                
            hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                orientation=np.radians(60),  # Rotate hexagons to align with grid
                                facecolor=colour, alpha=0.2, edgecolor=edgecolor, linewidth=linewidth)
            ax[j].text(x, y, text,  ha='center', va='center', size=15)  # Start numbering from 1
            ax[j].add_patch(hex)
            # Also add scatter points in hexagon centres
        ax[j].scatter(hcoord, vcoord, alpha=0, c = 'grey')
        ax[j].set_title(title)
        ax[j].set_aspect('equal')
    output_folder = derivatives_base/"analysis"/"maze_behaviour"
    output_folder.mkdir(exist_ok = True)
    output_file= output_folder / "proportion_correct.png"
    plt.savefig(output_file, dpi=300)
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"File saved to output folder: {output_folder}") 

        
def get_coords():   
    # Generate coordinates for a large hexagon with radius 4
    radius = 4
    coord = hex_grid(radius)

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartesian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coord]

    theta = np.radians(270)

    # Rotate the coordinates
    hcoord_rotated = [x * np.cos(theta) - y * np.sin(theta) for x, y in zip(hcoord, vcoord)]
    vcoord_rotated = [x * np.sin(theta) + y * np.cos(theta) for x, y in zip(hcoord, vcoord)]
            
    return hcoord, vcoord, hcoord_rotated, vcoord_rotated
    
def hex_grid(radius):
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

if __name__ == "__main__":
    derivatives_base = r"D:\Spatiotemporal_task\derivatives\sub-003_id_2V\ses-02_testHCT\test"
    rawsession_folder = r"D:\Spatiotemporal_task\rawdata\sub-003_id_2V\ses-02_date-05092025"
    plot_propcorrect(derivatives_base, rawsession_folder, [48, 29])