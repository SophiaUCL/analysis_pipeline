import numpy as np
from matplotlib.path import Path as MplPath
import os
import json
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from typing import Literal
from pathlib import Path
from scipy.spatial import ConvexHull

OverlayMethod = Literal["video", "image"] # methods for overlaying the hexagon grid. From the video or from an image in the docs
NumPlats = Literal[61, 127]


def calculate_cartesian_coords(coord: list [float], hex_side_length: int) -> tuple[list[float], list[float]]:
    """ Obtains horizontal and vertical (hcoord and vcoord) coordinates from
    axial coordinates, given side length of hex_side_length"""
    
    hcoord = [hex_side_length * c[0] * 1.5 for c in coord]  # Horizontal: scaled by 1.5 * side length
    vcoord = [hex_side_length * np.sqrt(3) * (c[1] - c[2]) / 2.0 for c in coord]  # Vertical: scaled
    return hcoord, vcoord

def translate_coords(hcoord2: list[float], vcoord2: list[float], theta: float, x_center: int,y_center: int, num_plats: NumPlats
) -> tuple[list[float], list[float]]:
    """ Translated and rotates coordinates
    hcoord2, vcoord2: horizontal and vertical coordinates of platforms
    theta: rotation angle
    x_center, y_center: x and y position of the middle platform (platform 31). Used to translate the coordinates"""
    hcoord_rotated = [x * np.cos(theta) - y * np.sin(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [x * np.sin(theta) + y * np.cos(theta) for x, y in zip(hcoord2, vcoord2)]
    vcoord_rotated = [-v for v in vcoord_rotated]


    # Calculate the translation needed to align the first rotated coordinate
    center = 30 if num_plats == 61 else 63
    dx = x_center - hcoord_rotated[center]
    dy = y_center - vcoord_rotated[center]

    # Apply the translation
    hcoord_translated = [x + dx for x in hcoord_rotated]
    vcoord_translated = [y + dy for y in vcoord_rotated]
    return hcoord_translated, vcoord_translated
    
def get_coordinates(params: dict) -> tuple[list[float], list[float]]:
    """ Gets x and y coordinates of platforms based on the params file"""
    radius = params['radius']
    hex_side_length = params['hex_side_length']
    theta = params['theta_angle']
    x_center = params['x_center']
    y_center = params['y_center']
    coord = hex_grid(radius)
    
    hcoord2, vcoord2 = calculate_cartesian_coords(coord, hex_side_length)
    
    hcoord_translated, vcoord_translated = translate_coords(hcoord2, vcoord2, theta, x_center, y_center)
    return hcoord_translated, vcoord_translated
    
    
    
def hex_grid(radius: int) -> list:
    """ Gives coordinates for hexagons for a maze with radius radius
    Returns them in axial coordinates (see https://www.redblobgames.com/grids/hexagons/ for further reading)"""
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append([q, r, -q - r])
    return coords


def is_point_in_platform(rat_locx: int, rat_locy: int, x_plat: int, y_plat: int, hex_side_length: int) -> bool:
    """ Checks whether the rat (rat_locx, rat_locy) on a given platform (x_plat, y_plat)
    if the hex side length is hex_side_length
    
    Used in get_platform_number
    """
    hex_vertices = []
    for angle in np.linspace(0, 2 * np.pi, num=6, endpoint=False):
        hex_vertices.append([
            x_plat + hex_side_length * np.cos(angle),
            y_plat + hex_side_length * np.sin(angle)
        ])
    hexagon_path = MplPath(hex_vertices)
    return hexagon_path.contains_point((rat_locx, rat_locy))

# Updated code to find the hexagon the rat is in
def get_platform_number(rat_locx: int, rat_locy: int, hcoord: list[float], vcoord: list[float], hex_side_length: int):
    """ Checks which platform a rat is on if its on position rat_locx, rat_locy
    hcoord and vcoord provide the horizontal and vertical platform coordinates
    hex_side_length gives the length of a side of a platform
    
    If a platform can't be found, nan is returned"""
    for i, (x, y) in enumerate(zip(hcoord, vcoord)):
        if is_point_in_platform(rat_locx, rat_locy, x, y, hex_side_length):
            return i + 1
    return np.nan

def get_platform_center(platform: int, params: dict):
    """ Returns the center position of platform, given the parameters in params"""
    hcoord_translated, vcoord_translated = get_coordinates(params)
    return [hcoord_translated[platform-1], vcoord_translated[platform-1]]

def get_nearest_platform(rat_locx: int, rat_locy: int, hcoord: list[float], vcoord: list[float]) -> int:
    """ Finds the nearest platform to a point. If a point is in a platform, it returns that number"""
    platform = get_platform_number(rat_locx, rat_locy, hcoord, vcoord)
    if not np.isnan(platform):
        return platform
    else:
        min_dist = 10**5
        closest_platform = 0
        for i, (x, y) in enumerate(zip(hcoord, vcoord)):
            dist = np.sqrt((rat_locx - x)**2 + (rat_locy - y)**2)
            if dist < min_dist:
                closest_platform = i + 1
        return closest_platform

def save_params(derivatives_base: Path, radius: int, hex_side_length: int, theta: int, desired_x: int, desired_y: int, rotation: int, hcoord_translated: list[float], vcoord_translated: list[float], pixels_per_cm: float = None, consinks: bool = False) -> None:
    """ Saves all params to derivatives\analysis\maze_overlay\maze_overlay_params.json
    if consinks = True, then it saves it to maze_overlay_params_consinks.json
    
    consinks = True is used when the function is called from overlay_maze_image_consinks, and it then has coordinates for 127 platforms, which 
    go past the maze aswell"""
    params = {
        "radius": radius,
        "hex_side_length": hex_side_length,
        "theta": theta,
        "x_center": desired_x,
        "y_center": desired_y,
        "rotation": rotation,
        "hcoord_tr": hcoord_translated,
        "vcoord_tr": vcoord_translated,
        "pixels_per_cm": pixels_per_cm
    }

    if not consinks:
        params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params.json")
    else:
        params_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_overlay_params_consinks.json")
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
        print(f"Parameters saved to: {params_path}")

def get_image(rawsession_folder: Path, method: OverlayMethod, tr: int = 1): 
    """ Gets image as either first video frame from the tracking folder 
    or the image from the code\config_files\image.png 
    
    tr denotes which trial video we overlay, defaults to 1"""
    
    # Getting image
    if method == "video":
        # Here it loads the first frame of the first video in the tracking folder
        pattern = f"T{tr}*.avi"
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

    
def makefig(angle: float, radius: int, hcoord_translated: list[float], vcoord_translated: list[float], img: np.ndarray, output_path = None, ratloc_x = None, ratloc_y = None, get_user_input: bool = True) -> bool:
    """ Overlay a hexagon (with coordinates hcoord_translated, vcoord_translated) onto img and saves it to output_path
    user shows whether overlay was good or not (y/n)
    
    if ratloc_x and ratloc_y is not None, it overlays that point on the image swell
    Returns
    -------
    approved: bool
        Whether the overlay was good or not
    """
    # Create the figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Display the image
    ax.imshow(img, cmap='gray')  
    
    
    # Overlay the hexagons
    for i, (x, y) in enumerate(zip(hcoord_translated, vcoord_translated)):
        hex = RegularPolygon((x, y), numVertices=6, radius=radius,
                             orientation=np.radians(angle),  # Rotate hexagons to align with grid
                             facecolor='none', alpha=1, edgecolor='y')
        ax.text(x, y, i + 1, ha='center', va='center', size=10, c = 'red')  # Start numbering from 1
        ax.add_patch(hex)
    
    # Add scatter points for hexagon centers (optional)
    ax.scatter(hcoord_translated, vcoord_translated, alpha=0, c='grey')
    
    if ratloc_x is not None and ratloc_y is not None:
        ax.scatter(ratloc_x, ratloc_y, c='blue', s=100, label='Rat Location')
    # Set limits to match the image dimensions
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)  # Flip y-axis for image alignment
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()
    
    if get_user_input:
        # User says whether overlay is good or not
        good_overlay = input('Enter whether overlay is good (y) or not (n): ')
        while good_overlay not in ['y', 'n']:
            print("Please input y or n")
            good_overlay = input('Enter input: ')
        plt.close()
        approved = good_overlay == "y"
        return approved
    else:
        return True



def get_params(derivatives_base: Path):
    """ Loads parameters from maze_overlay_params.json"""
    params_path = derivatives_base / "analysis" / "maze_overlay" / "maze_overlay_params.json"
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find {params_path}. Run overlay_maze_image() first.")

    with open(params_path, 'r') as f:
        params = json.load(f)

    hcoord = np.array(params["hcoord_tr"])
    vcoord = np.array(params["vcoord_tr"])
    side = params["hex_side_length"]
    rotation_deg = params["rotation"]
    
    return hcoord, vcoord, side, rotation_deg

def get_vertices(hcoord: list[float], vcoord: list[float], rotation_deg: float, side: float) -> list[float]:
    """ Computes the 6 vertices location for all of the platforms """

    # This creates an array that the location of the 6 vertices for each platform, scaled to each edge is length side
    base_hex = np.array([
        [np.cos(np.radians(0)),   np.sin(np.radians(0))],
        [np.cos(np.radians(60)),  np.sin(np.radians(60))],
        [np.cos(np.radians(120)), np.sin(np.radians(120))],
        [np.cos(np.radians(180)), np.sin(np.radians(180))],
        [np.cos(np.radians(240)), np.sin(np.radians(240))],
        [np.cos(np.radians(300)), np.sin(np.radians(300))]
    ]) * side

    all_vertices = []
    rotation_rad = np.radians(rotation_deg)
    
    # This is the rotation matrix, to rotate the base_hex by rotation_rad
    R = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad),  np.cos(rotation_rad)]
    ])
    
    # To each center point of the platform, we add the location of the vertices
    for x_c, y_c in zip(hcoord, vcoord):
        rotated_hex = base_hex @ R.T
        hex_vertices = rotated_hex + np.array([x_c, y_c])
        all_vertices.append(hex_vertices)

    all_vertices = np.vstack(all_vertices)
    return all_vertices

def get_outline(all_vertices: list[float]) -> tuple[list[float], list[float]]:
    """ Computer the complex hull of the vertices to get the outline"""
    hull = ConvexHull(all_vertices)
    hull_points = np.append(hull.vertices, hull.vertices[0])  # close loop
    outline_x = all_vertices[hull_points, 0].tolist()
    outline_y = all_vertices[hull_points, 1].tolist()
    return outline_x, outline_y

def save_outline(derivatives_base: Path, outline_x: list[float], outline_y: list[float]) -> None:
    """ Saves outline x and outline y to derivatives_base/analysis/maze_overlay/maze_outline_coords.json"""
    outline_path = derivatives_base / "analysis" / "maze_overlay" /  "maze_outline_coords.json"
    os.makedirs(os.path.dirname(outline_path), exist_ok=True)
    with open(outline_path, "w") as f:
        json.dump({"outline_x": outline_x, "outline_y": outline_y}, f, indent=4)
    print(f"Outline coordinates saved to: {outline_path}")

def plot_outline(derivatives_base: Path,img: np.ndarray,  outline_x: list[float], outline_y: list[float], hcoord: list[float], vcoord: list[float]) -> None:
    """ Overlays maze outline on img and saves it to derivatives_base / analysis / maze_overlay / maze_outline.png"""
    fig, ax = plt.subplots(figsize=(10, 10))
    if img is not None:
        ax.imshow(img, cmap="gray")
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

    ax.plot(outline_x, outline_y, "r-", lw=2, label="Maze outline")
    ax.scatter(hcoord, vcoord, color="yellow", s=10, label="Platform centers")

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Maze Outer Hexagon Outline (Edges)")
    
    output_path = os.path.join(derivatives_base, "analysis", "maze_overlay", "maze_outline.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"Maze outline figure saved to: {output_path}")