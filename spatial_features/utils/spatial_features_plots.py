import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes 
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

cmap_Cr = ListedColormap(["#384682", "#578da1", "#62b378", "#e6dc77", '#f0754d'])
def plot_rmap(rmap: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float, x_edges: np.ndarray, y_edges: np.ndarray, outline_x: np.ndarray | None, outline_y: np.ndarray | None, ax: Axes, fig: Figure, title: str = None) -> None:
    """ Plots ratemap"""

    im = ax.imshow(rmap.T, 
                cmap=cmap_Cr, 
                interpolation = None,
                origin='lower', 
                aspect='auto', 
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    if title is not None:
        ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect('equal')
    if outline_x is not None and outline_y is not None:
            ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    fig.colorbar(im, ax=ax, label='Firing rate')

def plot_spikes_spatiotemp(spike_train_this_epoch, x, y, epoch_end, frame_rate, xmin, xmax, ymin, ymax, ax, title = None):
    """ Creates scatterplot with all spikes until now (black) and spikes this epoch (red)"""
    x_until_now = x[:np.int32(epoch_end*frame_rate)]
    y_until_now = y[:np.int32(epoch_end*frame_rate)]
    ax.scatter(x_until_now, y_until_now, color = 'black', s= 0.7)
    if len(spike_train_this_epoch) > 0:
        ax.scatter(x[spike_train_this_epoch], y[spike_train_this_epoch], color = 'r', s= 0.7)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_aspect('equal', adjustable='box')
    if title is not None:
        ax.set_title(title)
    
def plot_occupancy(x: np.ndarray, y: np.ndarray,  xmin: float, xmax: float, ymin: float, ymax: float, outline_x: np.ndarray | None, outline_y: np.ndarray | None,  frame_rate: int, ax: Axes, fig: Figure):
    """ Plots the animal occupancy. Doesn't visualise bins that have occupancy above mean occupancy +stdv"""

    # Remove nans
    x_no_nan =  x[~pd.isna(x)]
    y_no_nan = y[~pd.isna(y)]
    
    # Bin spatial data
    heatmap_data, _, _  = np.histogram2d(x_no_nan, y_no_nan, bins=30, range=[[xmin, xmax], [ymin, ymax]])
    heatmap_data = heatmap_data/frame_rate

    # Get mean and stdv for points with occupancy above 0
    hm_nozero = heatmap_data[heatmap_data != 0]
    mean_hm = np.mean(hm_nozero)
    std_hm = np.std(hm_nozero) 
    
    # Threshold occupancy as to only show occupancy above threshold
    threshold = mean_hm + std_hm
    heatmap_data[heatmap_data > threshold] = 0

    # visualize
    im = ax.imshow(
            heatmap_data.T,
            cmap=cmap_Cr,
            interpolation=None,
            origin='upper',
            aspect='auto',
            extent=[xmin, xmax, ymax, ymin]
        )
    fig.colorbar(im, ax=ax, label='Seconds')
    if outline_x is not None and outline_y is not None:
        ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    ax.set_title('Occupancy full session')
    ax.set_aspect('equal')
    
def plot_directional_firingrate(bin_centers: np.ndarray, direction_firing_rate: np.ndarray, ax: Axes, title: str = None, MRL: float = None, percentiles_95_value: float = None):
    """ PLots directional firing rate"""
    # Plotting
    width = np.diff(bin_centers)[0]
    ax.bar(
        bin_centers,
        direction_firing_rate,
        width=width,
        bottom=0.0,
        alpha=0.8
    )
    if MRL is not None and percentiles_95_value is not None:
        if MRL > percentiles_95_value:
            text = f"MRL = {MRL:.2f}*"
        else:
            text = f"MRL = {MRL:.2f}"
        ax.text(
            np.pi/3,                # angle in radians
            np.nanmax(direction_firing_rate)*1.25,         # radius (just outside the bar)
            text,   # label text
            ha='center',
            va='bottom',
            fontsize=8,
            rotation_mode='anchor',
            color = 'r',
        )
    if title is not None:
        ax.set_title(title)

def plot_roseplots(filtered_df: pd.DataFrame,behaviour_df: pd.DataFrame, arms_dir: np.ndarray | list, arms_angles_start: np.ndarray | list, sum_count_bin: np.ndarray, bin_edges: np.ndarray,e: int,  tr: int, ax: Axes) -> None:
    """ Makes one subplot for roseplot"""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = np.deg2rad(bin_centers)
    width = np.diff(bin_centers)[0]

    # bar length = sum of number of spikes
    ax.bar(
        bin_centers,
        sum_count_bin,
        width=width,
        bottom=0.0,
        alpha=0.8,
        zorder = 2
    )

    ax.text(
        np.pi/3,                # angle in radians
        1.25* np.nanmax(sum_count_bin),         # radius (just outside the bar)
        f"n = {len(filtered_df)}",   # label text
        ha='center',
        va='bottom',
        fontsize=8,
        rotation_mode='anchor',
        color = 'r',
    )

    # Overlay the arm choices
    if e > 1:
        arm = behaviour_df.iloc[tr-1, 1]
        index = np.where(np.array(arms_dir) == arm)[0][0]
        angle_start = arms_angles_start[index]
        theta = np.linspace(np.deg2rad(angle_start), np.deg2rad(angle_start + 60), 100) 
        r = np.ones_like(theta) * np.nanmax(sum_count_bin)
        ax.fill_between(theta, 0, r, color='lightgreen', alpha=0.5, zorder=0)

    if e > 2:
        arm = behaviour_df.iloc[tr-1, 2]
        index = np.where(np.array(arms_dir) == arm)[0][0]
        angle_start = arms_angles_start[index]
        theta = np.linspace(np.deg2rad(angle_start), np.deg2rad(angle_start + 60), 100) 
        r = np.ones_like(theta) * np.nanmax(sum_count_bin)
        ax.fill_between(theta, 0, r, color='pink', alpha=0.5, zorder=0)


def add_arm_overlay_roseplot(behaviour_df: pd.DataFrame, tr:int, trials_to_include: np.ndarray | list, ax: Axes, fig: Figure) -> None:
    """ Overlays whether arm was correct or not"""
    if behaviour_df.iloc[tr-1, 3] == "Y":
        text = "Correct"
        c = 'g'
    else:
        text = "Incorrect"
        c = 'r'

    ax.remove() 
    ax = fig.add_subplot(len(trials_to_include), 5, (tr-1)*5 + 5) 
    ax.axis('off') 
    ax.text(0.0, 0.5, text, fontsize=11, va='center', ha='left', wrap=True, c= c)
    

def plot_spikemap_interactive_rmap(spike_train: np.ndarray | list, x: np.ndarray, y: np.ndarray, hd: np.ndarray, is_filt: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float, ax: Axes, outline_x = None, outline_y = None):
    """ Used in interactive rmap plot"""
    
    x_spikes = x[spike_train]
    y_spikes = y[spike_train]
    hd_spikes = hd[spike_train]


    valid = ~np.isnan(x_spikes) & ~np.isnan(y_spikes) & ~np.isnan(hd_spikes)
    x_spikes = x_spikes[valid]
    y_spikes = y_spikes[valid]
    hd_spikes = hd_spikes[valid]
    is_filt = is_filt[valid]  

    u = np.cos(hd_spikes)
    v = np.sin(hd_spikes)

    # Assign colors efficiently
    colors = np.where(is_filt, 'blue', 'red')

    # Plot
    ax.quiver(x_spikes, y_spikes, u, v, color=colors, scale = 30)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect('equal')
    if outline_x is not None and outline_y is not None:
            ax.plot(outline_x, outline_y, 'r-', lw=2, label='Maze outline')
    ax.set_title('Blue: within interval. Red: outside interval')
        
        