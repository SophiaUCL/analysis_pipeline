import numpy as np
import astropy.convolution as cnv

""" Ratemap functions"""
def get_ratemaps(spikes, x, y, n: int, binsize = 15, stddev = 5, frame_rate = 25):
    """
    Calculate the rate map for given spikes and positions.

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_no_nan = x[~np.isnan(x)]
    y_no_nan = y[~np.isnan(y)]
    
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    pos_binned, x_edges, y_edges = np.histogram2d(x_no_nan, y_no_nan, bins=[x_bins, y_bins])
    pos_binned = pos_binned/frame_rate
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_x_no_nan = spikes_x[~np.isnan(spikes_x)]
    spikes_y_no_nan = spikes_y[~np.isnan(spikes_y)]
    spikes_binned, _, _ = np.histogram2d(spikes_x_no_nan, spikes_y_no_nan, bins=[x_bins, y_bins])
    

    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    # Removing values with very low occupancy (these sometimes have very large firing rate)
    occupancy_threshold = 0.4
    rmap[smoothed_pos < occupancy_threshold] = np.nan

    return rmap, x_edges, y_edges


def get_ratemaps_restrictedx(spikes, x, y, x_restr, y_restr,  n: int, binsize = 15, stddev = 5, frame_rate = 25):
    """
    Calculate the rate map for given spikes and positions. x_restr and y_restr are used to calculate the occupancy map (since we're restricting over a time interval here)
    used for plot_rmap_interactive. For example used if you want to look only at one goal 

    Args:
        spikes (array): spike train for unit
        x (array): x positions of animal
        y (array): y positions of animal
        n (int): kernel size for convolution
        binsize (int, optional): binning size of x and y data. Defaults to 15.
        stddev (int, optional): gaussian standard deviation. Defaults to 5.

    Returns:
        rmap: 2D array of rate map
        x_edges: edges of x bins
        y_edges: edges of y bins
    """
    x_no_nan = x_restr[~np.isnan(x_restr)]
    y_no_nan = y_restr[~np.isnan(y_restr)]
    
    x_bins = np.arange(np.nanmin(x), np.nanmax(x) + binsize, binsize)
    y_bins = np.arange(np.nanmin(y), np.nanmax(y)+ binsize, binsize)

    # Pos: only for restricted data
    pos_binned, x_edges, y_edges = np.histogram2d(x_no_nan, y_no_nan, bins=[x_bins, y_bins])
    pos_binned = pos_binned/frame_rate
    spikes = [np.int32(el) for el in spikes]
    
    spikes_x = x[spikes]
    spikes_y = y[spikes]
    spikes_x_no_nan = spikes_x[~np.isnan(spikes_x)]
    spikes_y_no_nan = spikes_y[~np.isnan(spikes_y)]
    spikes_binned, _, _ = np.histogram2d(spikes_x_no_nan, spikes_y_no_nan, bins=[x_bins, y_bins])
    

    g = cnv.Gaussian2DKernel(stddev, x_size=n, y_size=n)
    g = np.array(g)
    smoothed_spikes =cnv.convolve(spikes_binned, g)
    smoothed_pos = cnv.convolve(pos_binned, g)

    rmap = np.divide(
        smoothed_spikes,
        smoothed_pos,
        out=np.full_like(smoothed_spikes, np.nan),  
        where=smoothed_pos != 0              
    )
    
    # Removing values with very low occupancy (these sometimes have very large firing rate)
    occupancy_threshold = 0.4
    rmap[smoothed_pos < occupancy_threshold] = np.nan
    return rmap, x_edges, y_edges
