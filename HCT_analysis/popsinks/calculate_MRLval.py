# relative directionFunction
import numpy as np
from astropy.stats import circmean
from population_sink.get_relDirDist import getRelDirDist

def mrlData(spikePos, spikeHD, spikePlats, relDirDist, binEdges, sink_bins):
    """
    Function to calculate the mrlData (originally, relativeDirectionFunction). 

    Args:
        spikePos (_type_): _description_
        spikeHD (_type_): _description_
        spikePlats (_type_): _description_
        relDirDist (_type_): _description_
        binEdges (_type_): edges of bin histogram that will be used (currently -pi to pi in 12 bins). Originally angleEdges
        sink_bins: bin locations, we extract xAxes and yAxes from it. 
    Returns:
        mrlData: dictionary with MRL values
    """
    # bincenters originally is calculated from angleEdges, but we won't use that

    xAxis = sink_bins['x']
    yAxis = sink_bins['y']
    print("Entered")
    totalDist_Rel = []

    totalDist_Rel = np.zeros_like(relDirDist[0])

    for p in np.arange(1,62): # 61 platforms
        # number of spikes for platform p
        nSpikesPerPlatform = spikePlats[p-1]

        if nSpikesPerPlatform == 0:
            continue # skip platform

        platDist_Rel = relDirDist[p-1] * nSpikesPerPlatform
        if len(totalDist_Rel)== 0:
            totalDist_Rel = platDist_Rel
        else:
            totalDist_Rel += platDist_Rel

    # They use a different function, but its basically the same
    dirRel2Goal_histCounts = getRelDirDist(spikePos, spikeHD, xAxis, yAxis, binEdges, normalize = False) 
    normDist = dirRel2Goal_histCounts/totalDist_Rel
    sumNormDist = normDist.sum(axis=2, keepdims=True)
    normDist *= len(spikeHD) / sumNormDist
    if np.isnan(normDist).any():
        print("Error, nan values in normDist")
        breakpoint()

    dir_bin_centres = (binEdges[1:] + binEdges[:-1])/2
    mrlData = mrlRelDir(normDist, xAxis, yAxis, dir_bin_centres) #should be bin centers?

    return mrlData


        
def mrlRelDir(histCounts, xAxis, yAxis, histBinCenters):

    mrl = np.zeros((histCounts.shape[0],histCounts.shape[1]))
    direction = np.full((histCounts.shape[0],histCounts.shape[1]), np.nan)

    for i in range(histCounts.shape[0]):
        for j in range(histCounts.shape[1]):
            weights = histCounts[i, j, :]
            mrl[i, j] = resultant_vector_length(histBinCenters, w = weights)
            direction[i,j] = circmean(histBinCenters, weights = weights)

    
    # Flatten
    mrl_Lin = mrl.ravel()
    
    # Finding max MRL value and its index
    mrl_max = np.nanmax(mrl_Lin)
    mrl_ind =  np.argmax(mrl_Lin)

    # Coordinates of maximum value
    mrl_MaxCoor_ind = np.unravel_index(mrl_ind, mrl.shape) # Check if order should be c?

    # Prefered direction at maximal MRL bin
    mrl_dir = direction[mrl_MaxCoor_ind[0], mrl_MaxCoor_ind[1]]
    mrl_dir_deg = np.rad2deg(mrl_dir)

    # MRL maximum coordinates
    mrl_MaxCoor = [xAxis[mrl_MaxCoor_ind[0]], yAxis[mrl_MaxCoor_ind[1]]]
                   
    # make dictionary
    mrlData = {'mrl': mrl_max,
               'coor': mrl_MaxCoor,
               'dir': mrl_dir,
               'dir_deg': mrl_dir_deg,
               'allMRL': mrl,
               'allDir': direction}
    return mrlData


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

