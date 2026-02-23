import numpy as np
import random
import warnings
import matplotlib.pyplot as plt

def get_sig_cells(spike_train_this_epoch: list, hd_rad: np.ndarray,epoch_start_frame: int, epoch_end_frame: int,  occupancy_time: np.ndarray | list, n_bins: int = 24, frame_rate: int = 25, num_shifts: int = 1000) -> tuple[float, float, list]:
    """
    Shuffles the data num_shifts time and calculates the MRLs

    Input:
    spike_train_this_epoch (list): spike data for unit from one epoch
    hd_rad (np.ndarray): hd array (for the whole trial and in radians!!!)
    epoch_start_frame (int): first frame of this epoch
    epoch_end_frame (int): last frame of this epoch
    occupancy_time (np.ndarray or list): histogram of occupancy per hd bin for this epoch
    num_bins (int): number of bins for the histogram (default = 24, so 15 degree bins)
    frame_rate (int): frame rate of video (default = 25)
    num_shifts (int): number of times data is shuffled. Default is 1000 
    
    Returns:
    perc_95_val: 95th percentile MRL value
    perc_99_val: 99th percentile MRL value
    MRL_values: All shuffled MRL values
    
    """
    # Setting shift values
    shift_min = 2*frame_rate # minimum shift: 2 second
    shift_max = np.int32(epoch_end_frame - epoch_start_frame) - 2*frame_rate # maximum shift: epoch length - 2 s
    
    if shift_min > shift_max:
        shift_max_temp = shift_min
        shift_min = shift_max
        shift_max = shift_max_temp
    if shift_min < 0:
        shift_min = 0

    MRL_values = []
    occupancy_time = np.nan_to_num(occupancy_time, nan=0.0)

    current_data = np.array(spike_train_this_epoch)
    shift_value = []
    
    max_radians = []
    val = 0
    
    range_min = np.int32(epoch_start_frame)
    range_max = np.int32(epoch_end_frame)
    range_size = range_max - range_min 
    for shift_idx in range(num_shifts):
        # Get random shift value
        random_shift = random.randint(shift_min, shift_max)

        # Add or subtract the random_shift value from each element in the variable
        shifted_data = current_data + random_shift

        
        # Ensure shifted_data stays within the range [range_min, range_max]
        shifted_data = np.mod(shifted_data - range_min, range_size) + range_min

        if np.nanmax(shifted_data) > range_max or np.nanmin(shifted_data) < range_min:
            breakpoint()
        # Calculate angles_degrees and MRL
        angles_radians= hd_rad[shifted_data]
        mask = ~np.isnan(angles_radians)
        angles_radians= angles_radians[mask]

        counts, bin_edges = np.histogram(angles_radians, bins=n_bins,range = [-np.pi, np.pi] )

        direction_firing_rate = np.divide(counts, occupancy_time, out=np.full_like(counts, 0, dtype=float), where=occupancy_time!=0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        MRL = resultant_vector_length(bin_centers, w = direction_firing_rate)
        MRL_values.append( MRL)
        shift_value.append(random_shift)
        if MRL > val:
            val = MRL
            max_radians = angles_radians

    perc_95_val = np.percentile(MRL_values, 95)
    perc_99_val = np.percentile(MRL_values, 99)
    if False:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Number of bins for angular resolution
        n_bins = 20
        ax.hist(max_radians, bins=n_bins, alpha=0.7)

        ax.set_title("Polar plot of max_radians = angles_radians")
        plt.show()
    return perc_95_val, perc_99_val, MRL_values


def resultant_vector_length(alpha, w=None, d=None, axis=None,
                            axial_correction=1, ci=None, bootstrap_iter=None):
    """
    Copied from Pycircstat documentation
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis,
                          axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r


def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    # Copied from picircstat documentation
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, "Dimensions of data " + str(alpha.shape) \
                                   + " and w " + \
        str(w.shape) + " do not match!"

    return ((w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) /
            np.sum(w, axis=axis))