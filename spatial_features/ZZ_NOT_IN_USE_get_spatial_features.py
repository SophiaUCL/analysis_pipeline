import numpy as np
from astropy.convolution import convolve, convolve_fft
from skimage.draw import disk



def getAdaptiveMap(pos_binned, spk_binned, alpha=200/30):
        """
        Produces a ratemap that has been adaptively binned according to the
        algorithm described in Skaggs et al., 1996) [1]_.

        Parameters
        ----------
        pos_binned : np.ndarray
            The binned positional data.
        spk_binned : np.ndarray
            The binned spikes
        alpha : int, optional
            A scaling parameter determing the amount of occupancy to aim at
            in each bin. Defaults to 200/30. In the original paper this was set to 200.
            This is 200/30 here as the pos data is binned in seconds (the original data was in pos
            samples so this is a factor of 30 smaller than the original paper's value, given 30Hz sample rate)

        Returns
        -------
        tuple of np.ndarray
            The adaptively binned spike and pos maps.
            Use this to generate Skaggs information measure

        Notes
        -----
        Positions with high rates mean proportionately less error than those
        with low rates, so this tries to even the playing field. This type
        of binning should be used for calculations of spatial info
        as with the skaggs_info method in the fieldcalcs class (see below)
        alpha is a scaling parameter that might need tweaking for different
        data sets.

        The data [are] first binned
        into a 64 X 64 grid of spatial locations, and then the firing rate
        at each point in this grid was calculated by expanding a circle
        around the point until the following criterion was met:

        Nspks > alpha / (Nocc^2 * r^2)

        where Nspks is the number of spikes emitted in a circle of radius
        r (in bins), Nocc is the number of occupancy samples, alpha is the
        scaling parameter
        The firing rate in the given bin is then calculated as:

        sample_rate * (Nspks / Nocc)

        References
        ----------
        .. [1] W. E. Skaggs, B. L. McNaughton, K. M. Gothard & E. J. Markus
            "An Information-Theoretic Approach to Deciphering the Hippocampal
            Code"
            Neural Information Processing Systems, 1993.
        """
        #  assign output arrays
        smthdpos = np.zeros_like(pos_binned)
        smthdspk = np.zeros_like(spk_binned)
        smthdrate = np.zeros_like(pos_binned)
        idx = pos_binned == 0
        pos_binned[idx] = np.nan
        spk_binned[idx] = np.nan
        visited = np.zeros_like(pos_binned)
        visited[pos_binned > 0] = 1
        # array to check which bins have made it
        bincheck = np.isnan(pos_binned)
        r = 1
        while np.any(~bincheck):
            # create the filter kernel
            h = disk(r)
            h[h >= np.max(h) / 3.0] = 1
            h[h != 1] = 0
            if h.shape >= pos_binned.shape:
                break
            # filter the arrays using astropys convolution
            filtpos = convolve(pos_binned, h)
            filtspk = convolve(spk_binned, h)
            filtvisited = convolve(visited, h)
            # get the bins which made it through this iteration
            truebins = alpha / (np.sqrt(filtspk) * filtpos) <= r
            truebins = np.logical_and(truebins, ~bincheck)
            # insert values where true
            smthdpos[truebins] = filtpos[truebins] / filtvisited[truebins]
            smthdspk[truebins] = filtspk[truebins] / filtvisited[truebins]
            bincheck[truebins] = True
            r += 1
        smthdrate = smthdspk / smthdpos
        smthdrate[idx] = np.nan
        smthdspk[idx] = np.nan
        smthdpos[idx] = np.nan
        return smthdrate, smthdspk, smthdpos

def skaggs_info(ratemap, dwelltimes, **kwargs):
    """
    Calculates Skaggs information measure

    Parameters
    ----------
    ratemap, dwelltimes :np.ndarray
        The binned up ratemap and dwelltimes. Must be the same size

    Returns
    -------
    float
        Skaggs information score in bits spike

    Notes
    -----
    The ratemap data should have undergone adaptive binning as per
    the original paper. See getAdaptiveMap() in binning class

    The estimate of spatial information in bits per spike:

    .. math:: I = sum_{x} p(x).r(x).log(r(x)/r)
    """
    sample_rate = kwargs.get("sample_rate", 30)

    dwelltimes = dwelltimes / sample_rate  # assumed sample rate of 30Hz
    if ratemap.ndim > 1:
        ratemap = np.reshape(ratemap, (np.prod(np.shape(ratemap)), 1))
        dwelltimes = np.reshape(dwelltimes, (np.prod(np.shape(dwelltimes)), 1))
    duration = np.nansum(dwelltimes)
    meanrate = np.nansum(ratemap * dwelltimes) / duration
    if meanrate <= 0.0:
        bits_per_spike = np.nan
        return bits_per_spike
    p_x = dwelltimes / duration
    p_r = ratemap / meanrate
    dum = p_x * ratemap
    ind = np.nonzero(dum)[0]
    bits_per_spike = np.nansum(dum[ind] * np.log2(p_r[ind]))
    bits_per_spike = bits_per_spike / meanrate
    return bits_per_spike

def coherence(smthd_rate, unsmthd_rate):
    """
    Calculates the coherence of receptive field via correlation of smoothed
    and unsmoothed ratemaps

    Parameters
    ----------
    smthd_rate : np.ndarray
        The smoothed rate map
    unsmthd_rate : np.ndarray
        The unsmoothed rate map

    Returns
    -------
    float
        The coherence of the rate maps
    """
    smthd = smthd_rate.ravel()
    unsmthd = unsmthd_rate.ravel()
    si = ~np.isnan(smthd)
    ui = ~np.isnan(unsmthd)
    idx = ~(~si | ~ui)
    coherence = np.corrcoef(unsmthd[idx], smthd[idx])
    return coherence[1, 0]
