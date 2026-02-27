import numpy as np
import os 
import pandas as pd


def calculate_relDirDist(pos_data, sink_bins, direction_bins):
    """ For each platform, has the relative directional distribution to each sink
    Saved in a 4D array (platform, sink x, sink y, direction bins)"""
    xAxis = sink_bins["x"]
    yAxis = sink_bins["y"]


    # We won't loop over the trial types, but they do
    # First they create an array with pos and HD and plat for each trial type
    # we already have this saved (forgot name), so we will not do this

    # from this we get plat, which is an array with the platform numbers
    # (for us, this is the column with platforms) in the pos csv
    # position has the xy position
    # hd has the hd
        
    plat = pos_data['platform']
    hd = pos_data['hd']
    position = pos_data.iloc[:, 0:2]
    
    # platform 1 to 61
    relDirDist = np.full((61, len(xAxis), len(yAxis), len(direction_bins) - 1),0, dtype = float)
    
    for p in np.arange(1, 62):

        platInd = np.where(plat == p)[0]

        if len(platInd) == 0:
            continue # skip if no dssata for this platform
        

        posPlat = position.iloc[platInd]
        hdPlat = hd[platInd]

        # Remove nan values
        valid_mask = ~np.isnan(hdPlat) & ~np.isnan(posPlat).any(axis=1)
        if not np.any(valid_mask):
            continue  # Skip if all NaNs for that platform

        posPlat = posPlat[valid_mask]
        hdPlat = hdPlat[valid_mask]

        relDirDistTemp = getRelDirDist(posPlat, hdPlat, xAxis, yAxis, direction_bins)
        relDirDist[p - 1, :, :, :] = relDirDistTemp 
    return relDirDist

def getRelDirDist(pos, hd, xAxis, yAxis, angleEdges, normalize = True):
    # Extract numeric columns as numpy arrays
    if isinstance(pos, pd.DataFrame):
        x = pos.iloc[:, 0].to_numpy()
        y = pos.iloc[:, 1].to_numpy()
    else:
        x = pos[:, 0]
        y = pos[:, 1]
    relDirDist = np.full((len(xAxis), len(yAxis), len(angleEdges) - 1), np.nan)
    for i, x_bin in enumerate(xAxis):
        for j, y_bin in enumerate(yAxis):

            xDistance = x_bin - x
            yDistance = y - y_bin

            dir2bin = np.atan2(yDistance, xDistance) ## CHECK IF THIS IS CORRECTT!!!!!

            dirRel2bin = hd - dir2bin
            dirRel2bin = (dirRel2bin + np.pi) % (2 * np.pi) - np.pi

            distTemp, _ = np.histogram(dirRel2bin, angleEdges)
            if normalize:
                distTemp = distTemp / distTemp.sum()
            relDirDist[i, j, :] = distTemp # format: x, y, dist
    return relDirDist
        

