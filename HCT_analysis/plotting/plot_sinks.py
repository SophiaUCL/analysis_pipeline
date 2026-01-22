import numpy as np
import os
import glob
import pandas as pd
from matplotlib.lines import Line2D
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from calculate_pos_and_dir import get_directions_to_position, get_relative_directions_to_position
from calculate_occupancy import get_relative_direction_occupancy_by_position, get_axes_limits, get_direction_bins, \
    bin_directions
from utilities.load_and_save_data import load_pickle, save_pickle
from utilities.restrict_spiketrain_specialbehav import restrict_spiketrain_specialbehav
from utilities.trials_utils import get_limits_from_json, get_goal_coordinates, get_coords
from matplotlib.patches import RegularPolygon
from matplotlib.patches import Patch

def plot_all_consinks(consinks_df, goal_coordinates, hcoord, vcoord, limits, jitter, plot_dir, plot_name='ConSinks'):
    num_goals = 3
    # Create two subplots
    fig, axes = plt.subplots(1, num_goals, figsize=(30, 10))
    axes = axes.flatten()
    fig.suptitle(plot_name, fontsize=24)

    for g in range(num_goals):
        ax = axes[g]
        ax.set_xlabel('x position (cm)', fontsize=16)
        ax.set_ylabel('y position (cm)', fontsize=16)

        # Add some coloured hexagons and adjust the orientation to match the rotated grid
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            hex = RegularPolygon((x, y), numVertices=6, radius=83,
                                 orientation=np.radians(28),  # Rotate hexagons to align with grid
                                 facecolor='grey', alpha=0.2, edgecolor='k')
            ax.text(x, y, i + 1, ha='center', va='center', size=15)  # Start numbering from 1
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c='grey')
        # plot the goal positions
        if g > 0:
            circle = plt.Circle(goal_coordinates[g - 1], 80, color='g',
                                fill=False, linewidth=5)
            ax.add_artist(circle)
            title = f'Goal {g}'
        else:
            title = 'G1 but going to G2'
        ax.set_title(title, fontsize=20)
        # loop through the rows of the consinks_df, plot a filled red circle at the consink
        # position if the mrl is greater than ci_999

        clusters = []
        consink_positions = []

        for cluster in consinks_df.index:

            x_jitter = np.random.uniform(-jitter[0], jitter[0])
            y_jitter = np.random.uniform(-jitter[1], jitter[1])

            consink_position = consinks_df.loc[cluster, 'position_g' + str(g)]

            sig = consinks_df.loc[cluster, 'sig_g' + str(g)]
            if sig == "sig":
                circle = plt.Circle((consink_position[0] + x_jitter,
                                     consink_position[1] + y_jitter), 60, color='r',
                                    fill=True)
                ax.add_artist(circle)

                clusters.append(cluster)
                consink_positions.append(consink_position)

            # set the x and y limits
            ax.set_xlim((limits['x_min'] - 200, limits['x_max'] + 200))
            ax.set_ylim(limits['y_min'] - 200, limits['y_max'] + 200)

            # reverse the y axis
            ax.invert_yaxis()

            # make the axes equal
            ax.set_aspect('equal')

    plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
    print(f"Saved figure to {os.path.join(plot_dir, plot_name + '.png')}")
    plt.show()

def plot_all_consinks_127sinks(consinks_df, goal_numbers, hcoord, vcoord, platforms_trans, jitter, plot_dir, average_sink = None, include_g0 = True, plot_name='ConSinks'):
    num_goals = 3
    # Create two subplots
    fig, axes = plt.subplots(1, num_goals - 1  + include_g0, figsize=(10*(num_goals - 1  + include_g0), 10))
    axes = axes.flatten()
    fig.suptitle(plot_name, fontsize=24)

    for g in range(num_goals):
        if g == 0 and not include_g0:
            continue
        ax = axes[g - 1 + include_g0]
        ax.set_xlabel('x position', fontsize=16)
        ax.set_ylabel('y position', fontsize=16)

        # Add some coloured hexagons and adjust the orientation to match the rotated grid
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            if i + 1 in platforms_trans:
                colour = 'grey'
                text = np.where(platforms_trans == i + 1)[0][0] + 1
                edgecolor = 'k'
                if g > 0 and goal_numbers[g-1] == text:
                    colour = 'green'
                    edgecolor = 'darkgreen'
            else:
                colour = 'white'
                text = " "
                edgecolor = 'white'
            hex = RegularPolygon((x, y), numVertices=6, radius=83,
                                 orientation=np.radians(28),  # Rotate hexagons to align with grid
                                 facecolor=colour, alpha=0.2, edgecolor=edgecolor)
            ax.text(x, y, text, ha='center', va='center', size=15)  # Start numbering from 1
            ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c='grey')
        # plot the goal positions

        # loop through the rows of the consinks_df, plot a filled red circle at the consink
        # position if the mrl is greater than ci_999

        clusters = []
        num_sig_clusters = 0
        mrl_vals = []
        for cluster in consinks_df.index:

            x_jitter = np.random.uniform(-jitter[0], jitter[0])
            y_jitter = np.random.uniform(-jitter[1], jitter[1])

            consink_plat = consinks_df.loc[cluster, 'platform_g' + str(g)]

            sig = consinks_df.loc[cluster, 'sig_g' + str(g)]
            if sig == "sig":
                num_sig_clusters += 1
                try:
                    circle = plt.Circle((hcoord[np.int32(consink_plat)-1] + x_jitter,
                                     vcoord[np.int32(consink_plat) - 1] + y_jitter), 30, color='r',
                                    fill=True)
                except:
                    breakpoint()
                ax.add_artist(circle)
                mrl_vals.append(consinks_df.loc[cluster, 'mrl_g' + str(g)])

                clusters.append(cluster)

        if average_sink[g] is not None:
            circle = plt.Circle(
                (average_sink[g][0], average_sink[g][1]),
                40,
                color='b',
                fill=True,
                label='Average sink'
            )

            ax.add_patch(circle)
            ax.legend()
        # make the axes equal
        ax.set_aspect('equal')

        # reverse the y axis
        ax.invert_yaxis()
        if g > 0:
            title = f'Goal {g}, num sig consinks: {num_sig_clusters}, mean mrl = {np.round(np.mean(mrl_vals),3)}'
        else:
            title = f'G1 but going to G2, num sig consinks: {num_sig_clusters}, mean mrl = {np.round(np.mean(mrl_vals),3)}'
        ax.set_title(title, fontsize=20)
        
    plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
    print(f"Saved figure to {os.path.join(plot_dir, plot_name + '.png')}")
    plt.show()

def plot_consinks_singlesubplot(consink_plat,consink_sig, max_mrls, mean_angles, percentile_95th, i_loc,  g, goal_numbers, hcoord, vcoord, platforms_trans,  whole_field_firing, jitter, title, ax):

    ax.set_xlabel('x position', fontsize=16)
    ax.set_ylabel('y position', fontsize=16)

    # Add some coloured hexagons and adjust the orientation to match the rotated grid
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if i + 1 in platforms_trans:
            colour = 'grey'
            text = np.where(platforms_trans == i + 1)[0][0] + 1
            edgecolor = 'k'
            if (g > 0  and g < 3 and goal_numbers[g - 1] == text) or (g == 3 and text in goal_numbers):
                colour = 'green'
                edgecolor = 'darkgreen'
        else:
            colour = 'white'
            text = " "
            edgecolor = 'white'
        hex = RegularPolygon((x, y), numVertices=6, radius=83,
                             orientation=np.radians(28),  # Rotate hexagons to align with grid
                             facecolor=colour, alpha=0.2, edgecolor=edgecolor)
        ax.text(x, y, text, ha='center', va='center', size=8)  # Start numbering from 1
        ax.add_patch(hex)

        # Also add scatter points in hexagon centres
        ax.scatter(hcoord, vcoord, alpha=0, c='grey')


    legend_handle = {}
    for i, plat in enumerate(consink_plat):
        x_jitter = np.random.uniform(-jitter[0], jitter[0])
        y_jitter = np.random.uniform(-jitter[1], jitter[1])

        if i == 0:
            colour = 'blue'
        else:
            colour = 'red'

        sig = consink_sig[i]

        x = hcoord[np.int32(plat) - 1] + x_jitter
        y = vcoord[np.int32(plat) - 1] + y_jitter

        if sig == "sig":
            marker = '*'
        else:
            marker = '.'
        ax.scatter(x, y, marker=marker, s=300, color=colour)

    # For the whole field
    x = hcoord[whole_field_firing['consink_plat']  -1]
    y = vcoord[whole_field_firing['consink_plat'] -1]
    if whole_field_firing['max_mrl'] > whole_field_firing['percentile_95th']:
        marker= '*'
    else:
        marker = '.'
    ax.scatter(x, y,marker = marker, s = 300, c='green')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = xmax - 500
    y = ymax - 250


    ax.text(x, y, f'MRL: {whole_field_firing['max_mrl']}', color='green')
    ax.text(x, y + 100, f'95th: {np.round(whole_field_firing['percentile_95th'], 3)}', color='green')
    ax.text(x, y + 200, f'{np.round(np.rad2deg(whole_field_firing['mean_angle']))}°', color='green')
    # make the axes equal
    legend_handles_color = [
        Patch(facecolor='blue', label='inside'),
        Patch(facecolor='red', label='outside'),
        Patch(facecolor='green', label='all spikes')
    ]
    legend_handles_type = [
        Line2D([0], [0], marker='*', color='k', linestyle='None', markersize=12, label='sig'),
        Line2D([0], [0], marker='.', color='k', linestyle='None', markersize=12, label='ns')
    ]


    if i_loc == 0:
        ax.legend(handles=legend_handles_color + legend_handles_type,
                  loc='upper right', fontsize = 8)

    colors = ['blue', 'red']
    for j in range(len(max_mrls)):
        x = xmin + 20
        y = ymin + 120 if j == 0 else ymax - 250
        ax.text(x, y, f'MRL: {max_mrls[j]}', color = colors[j])
        ax.text(x, y + 100, f'95th: {np.round(percentile_95th[j],3)}', color=colors[j])
        ax.text(x, y + 200, f'{np.round(np.rad2deg(mean_angles[j]))}°', color = colors[j])



    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title)

def plot_all_consinks_onegoal_127sinks(consinks_df, g, method,  goal_numbers, hcoord, vcoord, platforms_trans, jitter, average_sink = None, ax = None):

    ax.set_xlabel('x position', fontsize=16)
    ax.set_ylabel('y position', fontsize=16)

    # Add some coloured hexagons and adjust the orientation to match the rotated grid
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if i + 1 in platforms_trans:
            colour = 'grey'
            text = np.where(platforms_trans == i + 1)[0][0] + 1
            edgecolor = 'k'
            if g > 0 and goal_numbers[g-1] == text:
                colour = 'green'
                edgecolor = 'darkgreen'
        else:
            colour = 'white'
            text = " "
            edgecolor = 'white'
        hex = RegularPolygon((x, y), numVertices=6, radius=83,
                             orientation=np.radians(28),  # Rotate hexagons to align with grid
                             facecolor=colour, alpha=0.2, edgecolor=edgecolor)
        ax.text(x, y, text, ha='center', va='center', size=10)  # Start numbering from 1
        ax.add_patch(hex)

    # Also add scatter points in hexagon centres
    ax.scatter(hcoord, vcoord, alpha=0, c='grey')
    # plot the goal positions
    methods = ["trial_norm", "plat_norm"]
    if g > 0:
        title = f'Goal {g}, method {methods[method - 1]}'
    else:
        title = f'G1 but going to G2, method {methods[method - 1]}'
    ax.set_title(title, fontsize=20)
    # loop through the rows of the consinks_df, plot a filled red circle at the consink
    # position if the mrl is greater than ci_999

    clusters = []

    for cluster in consinks_df.index:

        x_jitter = np.random.uniform(-jitter[0], jitter[0])
        y_jitter = np.random.uniform(-jitter[1], jitter[1])

        consink_plat = consinks_df.loc[cluster, 'platform_g' + str(g)]

        sig = consinks_df.loc[cluster, 'sig_g' + str(g)]
        if sig == "sig":
            try:
                circle = plt.Circle((hcoord[np.int32(consink_plat)-1] + x_jitter,
                                 vcoord[np.int32(consink_plat) - 1] + y_jitter), 30, color='r',
                                fill=True)
            except:
                breakpoint()
            ax.add_artist(circle)

            clusters.append(cluster)

    if average_sink[g] is not None:
        circle = plt.Circle(
            (average_sink[g][0], average_sink[g][1]),
            40,
            color='b',
            fill=True,
            label='Average sink'
        )

        ax.add_patch(circle)
        ax.legend()
    # make the axes equal
    ax.set_aspect('equal')

    # reverse the y axis
    ax.invert_yaxis()

