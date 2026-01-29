import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt
import json 

def get_image(rawsession_folder, method="video"): 
    """ Gets image as either first video frame from the tracking folder 
    or the image from the code\config_files\image.png """
    
    # Getting image
    if method == "video":
        # Here it loads the first frame of the first video in the tracking folder
        pattern = "*T*.avi"
        files = glob.glob(os.path.join(rawsession_folder, 'tracking', pattern)) # finds matches
        video_path = files[0] # takes first video
        
        # reads video
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()   
        if not ret:
            print("Failed to read first frame")
        cap.release()
    elif method == "image":
        cwd = os.getcwd()
        #cwd = os.path.dirname(cwd) #Uncomment if running script from this code
        config_folder = os.path.join(cwd, "config_files")
        img_path = os.path.join(config_folder, "camera_image.png")
        if not os.path.exists(img_path):
            raise FileExistsError("Img path not found")
        img = cv2.imread(img_path)
    else:
        raise ValueError("Method not image or video. PLease provide valid input")
    return img


def save_limits(xmin, xmax, ymin, ymax, derivatives_base):
    """ Ensures xmin < xmax and ymin < ymax, and saves limits to
    derivatives_Base\analysis\maze_overlay\limits.json"""
    
    if xmin > xmax:
        xmin_temp = xmin
        xmin = xmax
        xmax = xmin_temp
    if ymin > ymax:
        ymin_temp = ymin
        ymin = ymax
        ymax = ymin_temp
        
    print(f"Final limits: xmin={xmin:.1f}, xmax={xmax:.1f}, ymin={ymin:.1f}, ymax={ymax:.1f}")
        
    limits =  {'x_min': xmin, 'x_max': xmax, 'x_width': xmax - xmin,
            'y_min': ymin, 'y_max': ymax, 'y_height': ymax - ymin}

    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    os.makedirs(os.path.dirname(limits_path), exist_ok=True)
    with open(limits_path, 'w') as f:
        json.dump(limits, f, indent=4)
        
    print(f"Limits saved to {limits_path}")
    return limits

def plot_sink_bins(derivatives_base, img = None, method = "video", sink_bins = None):
    """ This function plots the xy bins based on the limits obtained"""
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # Getting image
    if img is None:
        img = get_image(rawsession_folder, method = method)
        
    limits = get_limits_from_json(derivatives_base)
    if sink_bins is None:
        x_bins, y_bins, n_bins = get_xy_bins(limits)
    else:
        x_bins = sink_bins['x']
        y_bins = sink_bins['y']
        n_bins = len(x_bins) * len(y_bins)
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Display the image
    ax.imshow(img, cmap='gray')  

    x_bins_scatter = np.repeat(x_bins, len(y_bins))
    y_bins_scatter = np.tile(y_bins, len(x_bins))
    ax.scatter(x_bins_scatter, y_bins_scatter, c = 'r')
    plt.title(f"Sink bins (number of bins = {n_bins})")
    plt.show()
        

def get_xy_bins(limits, n_bins=120):

    # get the x and y limits of the maze
    x_min = limits['x_min']
    x_max = limits['x_max']
    x_width = limits['x_width']

    y_min = limits['y_min']
    y_max = limits['y_max']
    y_height = limits['y_height']

    # we want roughly 100 bins
    pixels_per_bin = np.sqrt(x_width*y_height/n_bins)
    n_x_bins = int(np.round(x_width/pixels_per_bin)) # note that n_bins is actually one more than the number of bins
    n_y_bins = int(np.round(y_height/pixels_per_bin))

    # create bins
    x_bins_og = np.linspace(x_min, x_max, n_x_bins + 1)
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin
    
    y_bins_og = np.linspace(y_min, y_max, n_y_bins + 1)
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin

    return x_bins, y_bins, n_bins
        
def get_limits_from_json(derivatives_base):
    """Gets the xy limits from the json file created in the get_limits.py function"""
    limits_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.json")
    with open(limits_path) as json_data:
        limits = json.load(json_data)
        json_data.close()
    return limits
    
def get_limits(derivatives_base):
    """
    This function gets the limits for field on which the consinks will be placed. 
    These limits are also used as the limits for several spatial plots (for example rate maps)
    and thus this code also has to be run for the spatiotemporal task. 

    Args:
        derivatives_base: Path to derivatives folder
        
    Creates:
        derivatives_base\analysis\maze_overlay\limits.png - limits overlayed on maze image
        derivatives_base\analysis\maze_overlay\limits.json - json with limits
        
    """
    # get rawsession folder
    rawsession_folder = derivatives_base.replace(r"\derivatives", r"\rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # Output folder
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "limits.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load image
    img = get_image(rawsession_folder)
    
    # Default parameters
    height, width, channels = img.shape
    xmin = 400
    xmax = 2250
    ymin = 10
    ymax = 1980
    
    good_limits = False
    while not good_limits:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        ax.set_aspect('equal')
        
        # Display the image
        ax.imshow(img, cmap='gray')  

        # v and h lines show the outline of the limits
        ax.vlines(x = xmin + 1, ymin = ymin + 1, ymax = ymax, color = 'r')
        ax.vlines(x = xmax, ymin = ymin + 1, ymax = ymax, color = 'r')
        ax.hlines(y=ymin + 1, xmin=xmin + 1, xmax=xmax,  color='r')
        ax.hlines(y=ymax, xmin=xmin + 1, xmax=xmax,  color='r')
        
        print("If you want to define new limits, click on the top left and then the bottom right of it")
        print("If you're happy with the limits, press escape")
        clicked = plt.waitforbuttonpress(timeout=-1)  # -1 → wait indefinitely
        
        if clicked: 
            plt.savefig(output_path)
            plt.close(fig)
            good_limits = True
            print("ESC pressed — keeping current limits.")
            break

        xy_coordinates = plt.ginput(2, timeout=0)  # timeout=0 → wait indefinitely
        (xmin, ymin), (xmax, ymax) = xy_coordinates
        print(f"New limits: xmin={xmin:.1f}, xmax={xmax:.1f}, ymin={ymin:.1f}, ymax={ymax:.1f}")
        plt.close()
    
    limits = save_limits(xmin, xmax, ymin, ymax, derivatives_base)
    
    print("Plotting sink bins location")
    plot_sink_bins(derivatives_base, img = img)
    return img, limits

        

if __name__ == "__main__":
    #derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-003_id-2F\ses-01_date-17092025\all_trials"
    derivatives_base = r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-01_date-10092025\all_trials"    
