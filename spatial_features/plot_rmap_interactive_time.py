


from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from utils.spatial_features_utils import get_spike_train_frames,  get_ratemaps, get_ratemaps_restrictedx, get_outline, get_occupancy_time, get_limits, get_posdata
from utils.spatial_features_plots import plot_directional_firingrate, plot_rmap, plot_spikemap_interactive_rmap
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Literal

Tasks = Literal["hct", "spatiotemp"]

def plot_rmap_interactive(derivatives_base: Path , unit_id: int, task: Tasks, last_trial_open_field: bool = False, frame_rate: int = 25, sample_rate: int = 30000):
    """ 
    Interactively define a temporal subregion on a spikecount plot and examine how a neuron’s directional tuning and ratemap changes when restricted to that region.
    Plots
    -----
    Left: ratemap
    Middle: spikemap, showing spikes during the selected intervals of the trials (red) and outside (blue)
    Right: HD distribution restricted to temperal area 
    Bottom: Spikecount of neuron throughout the trials
    
    Inputs
    -------
    derivatives_base (Path): Path to derivatives folder
    unit_id (int): unit that we're looking at
    tasks (hct or spatiotemp): task that we're doing
    last_trial_open_field (bool: False): Whether the last trial is an open field trial
    frame_rate (int: 25): frame rate of camera 
    sample_rate (int: 30000): sample rate of recording
    
    """
    # Load data files
    rawsession_folder = Path(str(derivatives_base).replace('derivatives', 'rawdata')).parent
    
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
    spike_train_unscaled = sorting.get_unit_spike_train(unit_id=unit_id)
    spike_train_s = np.array(spike_train_unscaled)/sample_rate
    input = 'c'
    bin_length = 60
    hd_restrict = None
    
    # Get original ratemap
    rmap, x_edges, y_edges = get_ratemaps(spike_train, x, y, 3, binsize=36, stddev=25)
    
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    mask_x = np.isnan(x)
    mask_y = np.isnan(y)
    y_bin[mask_y] = -1
    x_bin[mask_x] = -1
    pos_data['x_bin'] = x_bin
    pos_data['y_bin'] = y_bin
    
    # Get the data with trials length
    path_to_df = rawsession_folder/'task_metadata'/'trials_length.csv'
    if not path_to_df.exists():
        raise FileExistsError('trials_length.csv doesnt exist')
    trial_length_df = pd.read_csv(path_to_df)
    
    goal1_endtimes = None
    inputs = None
    x_int = None
    x_restrict = None
    y_restrict = None
    if task == 'hct':
        print("HCT: adding goal times to spikecount over trials")
        trialday_path = rawsession_folder/'behaviour'/'alltrials_trialday.csv'
        trialday_df  = pd.read_csv(trialday_path)
        if len(trialday_df) != len(trial_length_df) - last_trial_open_field:
            raise ValueError("length alltrials_trialday.csv is not the same as length trials to include. Remove unneeded trials")
        else:
            goal1_endtimes = np.array(trialday_df['Goal 1 end'])
            
    while input != 'q':
        # Make plot
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])

        ax1 = fig.add_subplot(gs[0, 0])           # ratemap
        ax2 = fig.add_subplot(gs[0, 1])           # quiver plot
        ax3 = fig.add_subplot(gs[0, 2], polar=True)  # polar plot
        ax4 = fig.add_subplot(gs[1, :])           # bottom row spans all columns

        fig.subplots_adjust(
            wspace=0.35,  # horizontal spacing
            hspace=0.45   # vertical spacing
        )
                
        fig.suptitle(f"Unit {unit_id}", fontsize = 18, y = 0.95)

        # ====== Plot ratemap ======
        
        # Masking values
        if inputs is not None:
            spike_train_filt = []
            for s in spike_train:
                s_in_sec = s/frame_rate # Adjusting s to be in seconds
                for start, end in x_int:
                    if s_in_sec > start and s_in_sec < end:
                        spike_train_filt.append(s)
        else:
            spike_train_filt = spike_train
        
        # Get original ratemap
        if inputs is not None:
            rmap, x_edges, y_edges = get_ratemaps_restrictedx(spike_train_filt, x, y, x_restrict, y_restrict, 3, binsize=36, stddev=25)

        
        plot_rmap(rmap, xmin, xmax, ymin, ymax, x_edges, y_edges, outline_x, outline_y, ax = ax1, fig = fig, title = f"{len(spike_train_filt)}/{len(spike_train)}")

        # 2. spikemap
        is_filt = np.isin(spike_train, spike_train_filt)
        plot_spikemap_interactive_rmap(spike_train, x, y, hd, is_filt, xmin, xmax, ymin, ymax, ax = ax2, outline_x = outline_x, outline_y = outline_y)
        
        # 3 Plot HD
        # Getting the spike times and making a histogram of them
         
        spikes_hd = hd[spike_train_filt]
        spikes_hd = spikes_hd[~np.isnan(spikes_hd)]
        #spikes_hd_rad = np.deg2rad(spikes_hd)
        spikes_hd_rad = spikes_hd
        counts, bin_edges = np.histogram(spikes_hd_rad, bins=num_bins,range = [-np.pi, np.pi] )

    
        # Calculating directional firing rate
        if hd_restrict is None:
            direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        else:
            hd_filtered_r = hd_restrict[~np.isnan(hd_restrict)]
            #hd_filtered_r= np.deg2rad(hd_filtered_r)
            occupancy_counts_r, _ = np.histogram(hd_filtered_r, bins=num_bins, range = [-np.pi, np.pi])
            occupancy_time_r = occupancy_counts_r / frame_rate 
            direction_firing_rate = np.divide(counts, occupancy_time_r, out=np.full_like(counts, 0, dtype=float), where=occupancy_time_r!=0)
        # MRL adn significance
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        plot_directional_firingrate(bin_centers, direction_firing_rate, ax = ax3, title = f'n (in region): {len(spike_train_filt)}')

        # Spike count over time
        
        total_trial_length = 0
        for tr in range(1, len(trial_length_df + 1)):
            trial_row = trial_length_df[(trial_length_df.trialnumber == tr)]
            trial_length = trial_row.iloc[0, 2]
            total_trial_length += trial_length

        n_bins = total_trial_length/bin_length


        # Simulated adjacent trials
        trial_lengths = np.array(trial_length_df['trial length (s)'])
        trial_ends = np.cumsum(trial_lengths)
        trial_starts = np.concatenate(([0], trial_ends[:-1]))

        # Plot
        ax4.hist(spike_train_s, bins = np.int32(n_bins))
        # Vertical lines at trial boundaries
        for start in trial_starts[1:]:
            ax4.axvline(x=start, color='black', linestyle='--', linewidth=1)
        ax4.axvline(x=trial_ends[-1], color='black', linestyle='--', linewidth=1)

        # Get current y-axis limits
        ymin4, ymax4 = ax4.get_ylim()

        # Label position: slightly below the top of the y-axis
        label_y = ymax4

        # Place trial labels
        for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            if last_trial_open_field and i == len(trial_starts) - 1:
                break
            mid = (start + end) / 2
            ax4.text(mid, label_y, f'Trial {i+1}',
                    ha='center', va='top', fontsize=9, color='black')
            if goal1_endtimes is not None:
                ax4.axvspan(
                            start,
                            start + goal1_endtimes[i],
                            facecolor='lightblue',  # or 'lightblue'
                            alpha=0.5,
                            zorder = 0
                        )
        if x_int is not None:
            for start, end in x_int:
                ax4.axvspan(
                            start,
                            end,
                            facecolor = 'red',  # or 'lightblue'
                            alpha=0.5,
                            zorder = 0
                        )
        # Optional: adjust y-limit if you want more headroom
        ax4.set_ylim(ymin4, ymax4 * 1.05)
        ax4.set_xlim(0, np.max(trial_ends))
        # Axis labels and title
        ax4.set_xlabel("Time (seconds)")
        ax4.set_ylabel("Number of spikes per minute")

        plt.tight_layout()
        print("Select intervals (up to 10 in total)")
        inputs = plt.ginput(20)
        x_points =  [val[0] for val in inputs] # only getting x values (These correspond to time)
        x_points.sort()
        
        if len(inputs) %2 != 0: # uneven number of points selected
            print("Uneven number of points selected. Last point will be ignored for intervals")
            x_points = x_points[:-1]
            
        x_int = []
        for i in range(np.int32(len(x_points)/2)):
            x_int.append((x_points[2*i], x_points[2*i + 1]))

        mask_time = []
        for i in range(len(pos_data)):
            in_int = False
            for start, end in x_int:
                if i/frame_rate > start and i/frame_rate < end:
                    mask_time.append(True)
                    in_int = True
                    break
            if not in_int:
                mask_time.append(False)
        x_restrict = x[mask_time]
        y_restrict = y[mask_time]
        hd_restrict = hd[mask_time]
        
        plt.show(block=True)
        plt.pause(0.1)
        plt.close(fig)


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





if __name__ == "__main__":
    derivatives_base = r"E:\Honeycomb_task_1g\derivatives\sub-001_id-2H\ses-01_date-01282026\first_run_2801"
    unit_id = 228
    task= "hct"
    last_trial_open_field = True
    plot_rmap_interactive(Path(derivatives_base), unit_id, task = "hct", last_trial_open_field=last_trial_open_field)



