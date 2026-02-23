import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('TkAgg')

""" This script is called by find_occupancy_live_downstairs in order to visualize the start platforms and the occupancy"""
def plot_startplatforms(df, goal_platforms, goals_to_include, ax):
    """
    Shows a maze plot with all the start platforms
    """
    
    trial_numbers = df['trial_number'].unique()
    
    start_platforms = []
    for t in trial_numbers:
        df_trial = df[df['trial_number'] == t]
        start_platform = df_trial['start platform'].iloc[0]
        start_platforms.append(start_platform)
        

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
    ax.set_aspect("equal")
    

def plot_occupancy(df, goal_platform, goal, ax):
    """
    Shows a maze plot with the occupancy of each platform for goal 1 (left) goal 2 (middle) (if applicable) and all trials (right)
    low: red, high: green

    """
    hcoord, vcoord, hcoord_rotated, vcoord_rotated = get_coords()


    cmap = plt.cm.RdYlGn
    # Go through goals and all trials
    if goal == 1:
        df_g = df[df['goal'] == goal_platform]
        title = "Goal 1"
    elif goal == 2:
        df_g = df[df['goal'] == goal_platform]
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
        if goal < 3 and i + 1 == goal_platform:
                edgecolor = "blue"
                linewidth = 5
        else:
            edgecolor = "k"
            linewidth = 1
        text = occupancy[i] if occupancy[i] > 0 else " "
        hex = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                            orientation=np.radians(60),  # Rotate hexagons to align with grid
                            facecolor=colour, alpha=0.2, edgecolor=edgecolor, linewidth= linewidth)
        ax.text(x, y, text,  ha='center', va='center', size=15)  # Start numbering from 1
        ax.add_patch(hex)
        # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c = 'grey')
    ax.set_title(title)
    ax.set_aspect('equal')

        
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


def make_plot(df, goal_platforms, goal_numbers, title = None):
    ngoals = len(goal_numbers)
    n_cols = ngoals + 1

    fig, axs = plt.subplots(1, n_cols, figsize = (6*n_cols, 6))
    
    plot_startplatforms(df, goal_platforms, goal_numbers, ax = axs[0])
    
    for i, goal in enumerate(goal_numbers):
        ax = axs[i + 1]
        goal_platform = goal_platforms[goal - 1] if goal < 3 else None
        plot_occupancy(df, goal_platform, goal, ax)
    
    if title is not None:
        plt.suptitle(title)
    plt.show()