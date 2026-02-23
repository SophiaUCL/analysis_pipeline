Folder: spatial_features
Contains files to calculate HD distribution, ratemap, roseplots, etc etc

combine_autowv_ratemaps: called by spatial_processing_pipeline. Combines autocorrelogram + wv plot,rmap + hd plot, and spikecount over trials plot. Also contains combine_autowv-ratemaps_vectorfields,which also adds the vectorfields plot on the right hand side. Saves to
analysis/cell_characteristics/spatial_features/autowv_ratemap_combined/unit_{unit}.png
and
analysis/cell_characteristics/spatial_features/autowv_ratemap_vectorfields_combined/unit_{unit}.png

plot_ratemap_and_hd: main function to get the hd distribution and rmap per unit. PLots rmap (left), occupancy (middle) and relative hd distribution (right).
outputs to derivatives_base/analysis/cell_characteristics/spatial_features/ratemaps_and_hd

plot_ratemap_and_hd: plots the ratemaps and head direction restricted to each goal segment, the full trial, and the open field (if present)
Saves to derivatives_base/analysis/cell_characteristics/spatial_features/ratemaps_and_hd_allgoals

plot_rmap_interactive: Interactively define a spatial subregion on a ratemap and examine how a neuron’s directional tuning changes when restricted to that region. Does it for one unit at a time.s

plot_rmap_interactive_time: Interactively define a temporal subregion on a spikecount plot and examine how a neuron’s directional tuning and ratemap changes when restricted to that region.

ZZ_NOT_IN_USE_get_spatial_features: Code gotten from Robin in order to calculate Skagg's information and coherence. Not implemented, see Chiara's code for these measures. 


Subfolder: utils
Contains files containing smaller functions called by other functions and functions to plot

spatial_features_utils: contains functions used in other functions, including creating rmap

spatial_Features_plots: plotting functions used in other functions

restrict_spiketrain_specialbehav: restrict spiketrain to goals for HCT