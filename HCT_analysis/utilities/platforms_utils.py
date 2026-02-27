import numpy as np
import pandas as pd
from matplotlib.path import Path
import os

"""
Different utilities relating to finding platforms locations etc.
"""

def hex_grid(radius):
    # Finds coordinates of platforms on hexagonal grid
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

def calculate_cartesian_coords(coord, hex_side_length):
    # Calculates cartesian coordinates from hexagonal coordinates
    hcoord = [hex_side_length * c[0] * 1.5 for c in coord]  # Horizontal: scaled by 1.5 * side length
    vcoord = [hex_side_length * np.sqrt(3) * (c[1] - c[2]) / 2.0 for c in coord]  # Vertical: scaled
    return hcoord, vcoord

# Updated code to find the hexagon the rat is in
def get_platform_number(rat_locx, rat_locy, hcoord, vcoord):
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if is_point_in_platform(rat_locx, rat_locy, x, y):
            return i + 1
    return -1 

def get_platform_center(hcoord_translated, vcoord_translated, platform):
    return [hcoord_translated[platform-1], vcoord_translated[platform-1]]

def get_nearest_platform(rat_locx, rat_locy, hcoord, vcoord):
    platform = get_platform_number(rat_locx, rat_locy, hcoord, vcoord)
    if platform != -1:
        return platform
    else:
        min_dist = 10**5
        closest_platform = 0
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            dist = np.sqrt((rat_locx - x)**2 + (rat_locy - y)**2)
            if dist < min_dist:
                closest_platform = i + 1
                min_dist = dist
        return closest_platform

def is_point_in_platform(rat_locx, rat_locy, hcoord, vcoord, hex_side_length):
    hex_vertices = []
    for angle in np.linspace(0, 2 * np.pi, num=6, endpoint=False):
        hex_vertices.append([
            hcoord + hex_side_length * np.cos(angle),
            vcoord + hex_side_length * np.sin(angle)
        ])
    hexagon_path = Path(hex_vertices)
    return hexagon_path.contains_point((rat_locx, rat_locy))

    
    
    
def get_goals_coords(goals):
    return [get_platform_center(goals[0]), get_platform_center(goals[1])]

def calculate_occupancy_plats(pos_data, frame_rate = 25):
    platforms_occupancy = []
    for i in range(61):
        platforms_i = pos_data[pos_data['platform'] == i + 1]
        occupancy_i = len(platforms_i)/frame_rate
        platforms_occupancy.append(occupancy_i)
    return platforms_occupancy

def get_firing_rate_platforms(spike_train, pos_data, platform_occupancy):
    platforms = pos_data['platform']
    platforms_spk = platforms[spike_train]
    firing_rate = []
    for p in np.arange(1,62):
        # Filter platforms_spk for platform = p
        platform_p = platforms_spk[platforms_spk == p]

        # Compute firing rate
        if platform_occupancy[p-1] > 0:
            rate = len(platform_p)/platform_occupancy[p-1]
        else:
            rate = 0

        firing_rate.append(rate)

    return firing_rate

def get_hd_distribution(hd, num_bins):
    hd_bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    hd_hist, bin_edges = np.histogram(hd, bins=hd_bins)
    bin_centers = (hd_bins[:-1] + hd_bins[1:]) / 2

    return hd_hist, bin_centers

def get_hd_distr_allplats(pos_data, num_bins = 12):
    hd_distr_allplats = []
    for p in np.arange(1, 62):
        hd = pos_data['hd']
        platforms = pos_data['platform']
        hd = np.array(hd)
        platforms = np.array(platforms)
        
        if np.nanmax(hd) > 2*np.pi + 0.1:
            hd = np.deg2rad(hd)
        indices = np.where(platforms == p)
        
        if len(indices) == 0:
            hd_distr_allplats.append(np.zeros(num_bins))
            continue
        hd_p = hd[indices[0]]
        hd_p = hd_p[~np.isnan(hd_p)]  # Remove NaNs
        hd_distr, bin_centers = get_hd_distribution(hd_p, num_bins=num_bins)
        hd_distr_allplats.append(hd_distr)
    return hd_distr_allplats, bin_centers

    
def get_norm_hd_distr(spike_train, pos_data, hd_distr_allplats, num_bins = 12):
    hd = pos_data['hd']
    platforms = pos_data['platform']
    if np.nanmax(hd) > 2*np.pi + 0.1:
        hd = np.deg2rad(hd)
    
    hd = hd[spike_train]
    platforms = platforms[spike_train]

    hd = np.array(hd)
    platforms = np.array(platforms)
    norm_hd = []
    for p in np.arange(1,62):
        # Get indices where platforms = p
        indices = np.where(platforms == p)
        if len(indices[0]) == 0:
            norm_hd.append(np.zeros(num_bins))
            continue
        
        hd_p = hd[indices[0]]
        hd_p = hd_p[~np.isnan(hd_p)]
        hd_p_distr, _ = get_hd_distribution(hd_p, num_bins = num_bins)
        hd_p_norm = np.divide(hd_p_distr, hd_distr_allplats[p-1],
                      out=np.zeros_like(hd_p_distr, dtype=float),
                      where=hd_distr_allplats[p-1]!=0)
        norm_hd.append(hd_p_norm)


    return norm_hd


