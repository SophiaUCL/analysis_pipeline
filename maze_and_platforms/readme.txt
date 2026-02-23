Folder: maze_and_platforms

Contains:
Files that have to do with the overlay of the hexagons on the video, determining the potential sink positions.

Specific files:
find_hexagon.ipynb - allows you to interactively overlay a hexagon on a video frame, in order to check whether the overlay is good. 
IMPORTANT: our overlay needs to match the platforms when raised, whilst most platforms in the video will be lowered. Therefore quality of overlay needs to be based on the platform that is raised. 
Also has a function to determine which platform a point is in.

find_platforms.py - utility file that contains functions to get hexagon coordinates, find out whether a point is in a platform, get center of platforms, etc etc. 

get_limits.py: called by spatial_processing_pipeline. File that is used to find the limits of what we will plot, and in case of the honeycomb task, overlay sinks on it aswell (though we don't use these sinks anymore).
Outputs:
	derivatives_base\analysis\maze_overlay\limits.png - limits overlayed on maze image
        derivatives_base\analysis\maze_overlay\limits.json - json with limits


overlay_maze_image: overlays a hexagonal grid with parameters defined inside the function onto a frame of the video (if method = video) or on the image in the code files (if method = image). Outputs the parameters for the overlay. Used in the spatial_processing_pipeline function.
Outputs:
        Into derivatives/analysis/maze_overlay:
            maze_overlay.png: image with hex grid overlay
            maze_overlay_params.json: parameters used to create the overlay

overlay_maze_image_consinks: overlays a hexagonal grid of radius 8 (so 127 consinks) onto a frame of the video or on the image in the code files. This is used when we want our potential sink positions to match the hexagonal grid of our maze + extend a bit behind it. 
Outputs: Into derivatives/analysis/maze_overlay:
            maze_overlay_consinks.png: image with hex grid overlay
            maze_overlay_params_consinks.json: parameters used to create the overlay

visualize_rat_location: Overlays a point (ratloc_x, ratloc_y) onto a frame of the video of the trials, returns which platform it is on, and also overlays the hexagonal grid

overlay_maze_outline: Plots and saves the outer boundary of the honeycomb maze (edges of the outermost platforms), and saves outline coordinates to JSON.
Outputs
        derivatives_base\analysis\maze_overlay\maze_outline_coords.json
            outline coordinates for maze
        derivatives_base\analysis\maze_overlay\maze_outline.png
            image with maze outline visualisedW


Old parameters for video overlay:
    """ Params used before 01/26
    hex_side_length = 88 # Length of the side of the hexagon
    theta = np.radians(305)  # Rotation angle in radians
    desired_x, desired_y = 1320, 950 # center of 31st platform
    rotation = 25
    
    Params used for old camera
    hex_side_length = 89
    theta = np.radians(30)
    desired_x, desired_y = 810, 1000
    rotation = 50

    Params before 13/02 (so still used for 2H ses 1
radius = 4
    hex_side_length = 88 # Length of the side of the hexagon
    theta = np.radians(305)  # Rotation angle in radians
    desired_x, desired_y = 1340, 970 # center of 31st platform
    coord = hex_grid(radius) # coordinates
    rotation = 25

2H ses 2 onwards
radius = 4
    hex_side_length = 88 # Length of the side of the hexagon
    theta = np.radians(312)  # Rotation angle in radians
    desired_x, desired_y = 1318, 953 # center of 31st platform
    coord = hex_grid(radius) # coordinates
    rotation = 17

    """