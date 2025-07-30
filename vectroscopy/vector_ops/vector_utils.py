import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def save_gdf_to_file(gdf, output_path, file_name, driver='ESRI Shapefile'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = os.path.join(output_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    print(f"Saving GeoDataFrame to {file_path} with driver {driver}")
    gdf.to_file(file_path, driver=driver)


"""Color assignment utilities for GeoDataFrames"""
def assign_color(gdf, color="viridis"):
    """
    Assign colors to the geometries in the GeoDataFrame based on the thresholds.

    Args:
        gdf: The GeoDataFrame containing the geometries.
        colormap (str): The name of the matplotlib colormap to use for coloring the geometries.

    Returns:
        GeoDataFrame: The input GeoDataFrame with an added 'color' column.
    """
    c = {'red': [255,0,0], 'orange': [255,128,0], 'yellow': [255,255,0],
        'lime': [128,255,0], 'green': [0,255,0], 'sea': [0,255,128],
        'cyan': [0,255,255], 'sky': [0,128,255], 'blue': [0,0,255],
        'violet': [128,0,255], 'magenta': [255,0,255], 'pink': [255,0,128]}
    
    thresholds = gdf['Threshold'].unique()
    
    if color in c:
        end_color = c[color]
        # Use a linear color ramp from white to the selected color, mapped by threshold value
        thresholds_sorted = sorted(thresholds)
        color_map = make_ramp(end_color, len(thresholds_sorted))
        threshold_to_color = {val: color_map[i][0] for i, val in enumerate(thresholds_sorted)}
        gdf['hex_color'] = gdf['Threshold'].map(threshold_to_color)

    elif color in plt.colormaps():
        cmap = plt.get_cmap(color, len(thresholds))
        color_map = {val: mcolors.to_hex(cmap(i)) for i, val in enumerate(sorted(thresholds))}
        gdf['hex_color'] = gdf['Threshold'].map(color_map)
    
    else:
        raise ValueError(f"Color '{color}' is not recognized. Use a valid matplotlib colormap name or a predefined color.")

    return gdf

def make_ramp(end_color, num_thresh):
    r, g, b = end_color
    rs, gs, bs = (
        np.linspace(0, r, num_thresh+1),
        np.linspace(0, g, num_thresh+1),
        np.linspace(0, b, num_thresh+1))
    return [[rgb_to_hex(int(rs[i]), int(gs[i]), int(bs[i]))] for i in range(num_thresh+1)]

def rgb_to_hex(r, g, b):
    """
    Converts RGB color values (0-255) to a hexadecimal color code.

    Args:
        r (int): Red component (0-255).
        g (int): Green component (0-255).
        b (int): Blue component (0-255).

    Returns:
        str: The hexadecimal color code in the format '#RRGGBB'.
    """
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)
