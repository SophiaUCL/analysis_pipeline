import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import os
import cv2
import json
import glob

def get_image(rawsession_folder, method): 
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

def makefig(angle, radius, hcoord_translated, vcoord_translated, img, output_path):# Create the figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Display the image
    ax.imshow(img, cmap='gray')  
    
    
    # Overlay the hexagons
    for i, (x, y) in enumerate(zip(hcoord_translated, vcoord_translated)):
        hex = RegularPolygon((x, y), numVertices=6, radius=radius,
                             orientation=np.radians(angle),  # Rotate hexagons to align with grid
                             facecolor='none', alpha=1, edgecolor='y')
        ax.text(x, y, i + 1, ha='center', va='center', size=10)  # Start numbering from 1
        ax.add_patch(hex)
    
    # Add scatter points for hexagon centers (optional)
    ax.scatter(hcoord_translated, vcoord_translated, alpha=0, c='grey')
    
    # Set limits to match the image dimensions
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)  # Flip y-axis for image alignment
    plt.savefig(output_path)
    plt.show()
    good_overlay = input('Enter whether overlay is good (y) or not (n): ')
    while good_overlay not in ['y', 'n']:
        print("Please input y or n")
        good_overlay = input('Enter input: ')
    plt.close()
    return good_overlay

# Function to calculate Cartesian coordinates with scaling
def calculate_cartesian_coords(coord, hex_side_length):
    hcoord = [hex_side_length * c[0] * 1.5 for c in coord]  # Horizontal: scaled by 1.5 * side length
    vcoord = [hex_side_length * np.sqrt(3) * (c[1] - c[2]) / 2.0 for c in coord]  # Vertical: scaled
    return hcoord, vcoord


def hex_grid(radius):
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords

def get_translated_coords(hcoord2, vcoord2, theta, desired_x, desired_y):
    """ Takes hcoord2, vcoord2 and rotates them by theta and then translated them by
    desired_x and desired_y. Outputs hcoord_translated, vcoord_translated"""
    # Rotate the coordinates
    hcoord_rotated = [x * np.cos(theta) - y * np.sin(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [x * np.sin(theta) + y * np.cos(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [-v for v in vcoord_rotated]

    # Calculate the translation needed to align the first rotated coordinate
    dx = desired_x - hcoord_rotated[30]
    dy = desired_y - vcoord_rotated[30]

    # Apply the translation
    hcoord_translated = [x + dx for x in hcoord_rotated]
    vcoord_translated = [y + dy for y in vcoord_rotated]
    return hcoord_translated, vcoord_translated


def save_params(derivatives_base, radius, hex_side_length, theta, desired_x, desired_y, rotation, hcoord_translated, vcoord_translated):
    """ Saves all params to derivatives\analysis\maze_overlay\maze_overlay_params.json"""
    params = {
        "radius": radius,
        "hex_side_length": hex_side_length,
        "theta": theta,
        "x_center": desired_x,
        "y_center": desired_y,
        "rotation": rotation,
        "hcoord_tr": hcoord_translated,
        "vcoord_tr": vcoord_translated
    }

    params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
        print(f"Parameters saved to: {params_path}")

    
def overlay_maze_image(derivatives_base, method):
    """
    This code overlays a hexagonal grid on
    the image in the code files that has the maze with allplatforms up (if method == image)
    the first frame of the videos for this session (if method == video)
    In the code files, there's an image saved where all maze platforms are up. This function overlays the hex grid
    on top of that image to show the maze layout
    
    Outputs:
        Into derivatives/analysis/maze_overlay:
            maze_overlay.png: image with hex grid overlay
            maze_overlay_params.json: parameters used to create the overlay
    
    """
    # Loading rawsession folder
    rawsession_folder = derivatives_base.replace("derivatives", "rawdata")
    rawsession_folder = os.path.dirname(rawsession_folder)
    
    # Obtaining image used for overlay
    img = get_image(rawsession_folder, method)

    # Output folder
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Parameters that control the overlay
    radius = 4
    hex_side_length = 88 # Length of the side of the hexagon
    theta = np.radians(305)  # Rotation angle in radians
    desired_x, desired_y = 1340, 970 # center of 31st platform
    coord = hex_grid(radius) # coordinates
    rotation = 25
    #rotation = 50
    
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
    """
    # Calculate initial Cartesian coordinates
    hcoord2, vcoord2 = calculate_cartesian_coords(coord, hex_side_length)
    hcoord_translated, vcoord_translated = get_translated_coords(hcoord2, vcoord2, theta, desired_x, desired_y)
    
    # plots figure
    good_overlay = makefig(rotation,hex_side_length, hcoord_translated, vcoord_translated, img, output_path)
    
    # Save parameters to json file
    save_params(derivatives_base, radius, hex_side_length, theta, desired_x, desired_y, rotation, hcoord_translated, vcoord_translated)

    print(f"Maze overlay saved to {output_path}")

    if good_overlay == 'y':
        print("Overlay approved, continuing all operations")
    else:
        print("Overlay not approved. Adjust parameters in overlay_maze_image_fromVideo function")
        print('Spatial processing pipeline will assign platforms to positional csvs.\n ')
    return good_overlay, img

if __name__ == "__main__":
    derivatives_base =  r"S:\Honeycomb_maze_task\derivatives\sub-002_id-1R\ses-02_date-11092025\all_trials"
    overlay_maze_image(derivatives_base, "video")
