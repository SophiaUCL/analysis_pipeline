
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as MplPath
import numpy as np
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from utils.spatial_features_utils import get_spike_train_frames, get_ratemaps, get_limits, get_outline, get_posdata, get_occupancy_time
from utils.spatial_features_plots import plot_spikemap_interactive_rmap, plot_rmap,plot_directional_firingrate
from pathlib import Path
                    
    
def plot_rmap_interactive(derivatives_base: Path, unit_id: int,  frame_rate: int = 25, sample_rate: int = 30000):
    """ 
    Interactively define a spatial subregion on a ratemap and examine how a neuron’s directional tuning changes when restricted to that region.
    Plots
    -----
    Left: ratemap, which you can overlay a polygon on
    Center-left: spikemap, showing which area was selected (red)
    Center-right: HD distribution. HD spikes in the area divided by animal HD distribution over the whole trial
    Right: HD distribution. HD spikes in the area divided by animal HD distribution in that area
    
    Inputs
    -------
    derivatives_base (Path): Path to derivatives folder
    unit_id (int): unit that we're looking at
    frame_rate (int: 25): frame rate of camera 
    sample_rate (int: 30000): sample rate of recording
    
    """
    # Load data files
    kilosort_output_path = derivatives_base/ 'ephys'/ "concat_run"/"sorting"/ "sorter_output" 
    sorting = se.read_kilosort(
        folder_path = kilosort_output_path
    )
   
    # Limits and outline
    xmin, xmax, ymin, ymax = get_limits(derivatives_base)
    outline_x, outline_y = get_outline(derivatives_base)
        
    # Get directory for the positional data
    x, y, hd, pos_data = get_posdata(derivatives_base, method = "ears")
    
    
    # Loop over units
    print("Plotting ratemaps and hd")

    # Obtaining hd for this trial how much the animal sampled in each bin
    num_bins = 24
    occupancy_time = get_occupancy_time(hd, frame_rate, num_bins = num_bins)

    # Load spike data
    spike_train = get_spike_train_frames(sorting, unit_id, x, sample_rate, frame_rate)
    
    # Mask variables
    input = 'c'
    mask = None 
    
    # Get original ratemap
    rmap, x_edges, y_edges = get_ratemaps(spike_train, x, y, 3, binsize=36, stddev=25)
    
    # Binning pos_data
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    mask_x = np.isnan(x)
    mask_y = np.isnan(y)
    y_bin[mask_y] = -1
    x_bin[mask_x] = -1
    pos_data['x_bin'] = x_bin
    pos_data['y_bin'] = y_bin
     
    while input != 'q':
        # Make plot
        fig, axs = plt.subplots(1, 4, figsize = [20, 5])
        axs = axs.flatten()
        fig.suptitle(f"Unit {unit_id}", fontsize = 18)

        # ====== Plot ratemap ======
        if mask is not None:
            hd_masked = mask_posdata(pos_data, mask)

            occupancy_counts_masked, _ = np.histogram(hd_masked, bins=num_bins, range = [-np.pi, np.pi])
            occupancy_time_masked = occupancy_counts_masked / frame_rate 
    
            xsize, ysize = mask.shape
            spike_train_filt = []
            for s in spike_train:
                indx = x_bin[s]
                indy = y_bin[s]
                if indx < xsize and indy < ysize and mask[indx, indy]:
                    spike_train_filt.append(s)
        else:
            spike_train_filt = spike_train
            
        is_filt = np.isin(spike_train, spike_train_filt)

        # 1. PLot rmap
        plot_rmap(rmap, xmin, xmax, ymin, ymax, x_edges, y_edges, outline_x, outline_y, ax = axs[0], fig = fig, title = f"{len(spike_train_filt)}/{len(spike_train)}")

        # 2. spikemap
        plot_spikemap_interactive_rmap(spike_train, x, y, hd, is_filt, xmin, xmax, ymin, ymax, ax = axs[1], outline_x = outline_x, outline_y = outline_y)
        
        # 3 & 4.  Plot HD
        # Getting the spike times and making a histogram of them
        spikes_hd = hd[spike_train_filt]
        spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
        #spikes_hd_rad = np.deg2rad(spikes_hd)
        spikes_hd_rad = spikes_hd
        counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

        
        # Calculating directional firing rate
        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        if mask is not None:
            direction_firing_rate_masked = np.divide(counts, occupancy_time_masked,out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0 )
        else:
            direction_firing_rate_masked = direction_firing_rate
        fig.delaxes(axs[2])
        axs[2] = fig.add_subplot(1,4,3, polar=True)

        # MRL adn significance
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plot_directional_firingrate(bin_centers, direction_firing_rate, ax = axs[2], title = "HD in area, divided by animal HD distr in whole trial")
    
        # Plotting
        fig.delaxes(axs[3])
        axs[3] = fig.add_subplot(1,4,4, polar=True)
        plot_directional_firingrate(bin_centers, direction_firing_rate_masked, ax = axs[3], title = "HD in area, divided by animal HD distr in area")
        
        plt.tight_layout()
        plt.show(block=False)   # show figure but don’t block execution yet

        # Now let the user draw on the ratemap (axs[0])
        mask = select_region(rmap, axs[0], x_edges, y_edges)
        inside_coords = np.argwhere(mask)
        print("Number of points inside polygon:", inside_coords.shape[0])


def select_region(data, ax, x_edges, y_edges):
    """
    Allows the user to draw a polygon on the given Axes and returns a boolean mask
    for the region inside the polygon (same shape as `data`).
    """

    mask_container = {'done': False, 'mask': None}

    def onselect(verts):

        ny, nx = data.shape
        x_lin = np.linspace(x_edges[0], x_edges[-1], ny)  # ny instead of nx
        y_lin = np.linspace(y_edges[0], y_edges[-1], nx)  # nx instead of ny
        X, Y = np.meshgrid(x_lin, y_lin)

        points = np.vstack((X.ravel(), Y.ravel())).T
        path = MplPath(verts)
        mask = path.contains_points(points).reshape((nx, ny)).T  # transpose back
        mask_container['mask'] = mask
        mask_container['done'] = True

        ax.contour(mask, colors='r', linewidths=0.8)
        plt.draw()

        selector.disconnect_events()
        plt.close()

    selector = PolygonSelector(ax, onselect, props=dict(color='r', linewidth=2, alpha=0.6))

    print("Draw your polygon on the ratemap (double-click to close it)...")
    plt.show(block=True)

    # Wait until user finishes
    while not mask_container['done']:
        plt.pause(0.1)

    return mask_container['mask']


def mask_posdata(pos_data, mask):
    """
    Masks positional data

    Args:
        pos_data: array with x, y, hd, x_bin and y_bin
        mask: 2D boolean
    returns hd_masked
    """
    xsize, ysize = mask.shape

    x_bin = pos_data["x_bin"].to_numpy()
    y_bin = pos_data["y_bin"].to_numpy()
    hd = pos_data["hd"].to_numpy()

    valid = (
        (x_bin >= 0) & (x_bin < xsize) &
        (y_bin >= 0) & (y_bin < ysize)
    )

    valid_mask = np.zeros_like(valid, dtype=bool)
    valid_indices = np.where(valid)
    valid_mask[valid_indices] = mask[x_bin[valid], y_bin[valid]]

    # Return masked hd values (ignore NaNs)
    return hd[valid_mask & ~np.isnan(hd)]
          



if __name__ == "__main__":
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
    unit_id = 228
    plot_rmap_interactive(Path(derivatives_base), unit_id)


