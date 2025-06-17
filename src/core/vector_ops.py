import os
import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry import shape
import pandas as pd
from tqdm import tqdm
from rasterio.io import MemoryFile
from exactextract import exact_extract

def list_vectorize(raster_list, thresholds, crs, transform):
    """
    Vectorizes a list of rasters using a list of thresholds.
    """
    index = 0
    results = []
    for raster in tqdm(raster_list, desc="Vectorizing"):
        result = vectorize(raster, thresholds[index], transform, crs)
        results.append(result)
        index += 1
        
    return results

def vectorize(array, value=1, transform=None, crs=None):
    """
    Convert a NumPy array into a GeoDataFrame of polygons where array == value.

    Parameters:
    - array (np.ndarray): 2D array of data values.
    - value (int or float): Value in array to convert to polygons.
    - transform (Affine): Affine transform mapping array to coordinates.
    - crs (dict or str or CRS): Coordinate reference system.

    Returns:
    - GeoDataFrame: Polygon geometries corresponding to pixels == value.
    """

    if transform is None:
        # Assume 1 unit per pixel in upper-left origin if not specified
        transform = Affine.translation(0, 0) * Affine.scale(1, -1)

    # Extract shapes from the binary mask
    shapes_gen = features.shapes(array, transform=transform)

    # Create list of GeoJSON-like geometries
    geoms = [
        {"geometry": shape(geom), "value": value}
        for geom, val in shapes_gen if val == 1
    ]

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(geoms, crs=crs)

    return gdf

def merge_polygons(gdfs):
    """
    Merge a list of GeoDataFrames into a single GeoDataFrame.
    Returns an empty GeoDataFrame if the input list is empty.
    """
    if not gdfs:
        return gpd.GeoDataFrame()
    return pd.concat(gdfs, ignore_index=True)
    # merged = gdf.dissolve(by="value", tolerance=tolerance)
    # return merged.reset_index(drop=True)

def show_polygons(gdf, title=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf.plot(ax=ax, color='blue', edgecolor='black')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def save_shapefile(gdf, output_path, file_name, driver='ESRI Shapefile'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gdf.to_file(os.path.join(output_path, file_name), driver=driver)

def simplify_raster_geometry(gdf, tolerance):
    """
    Simplify polygons using the GeoSeries.simplify method.

    Parameters:
    - gdf (GeoDataFrame): Vectorized raster polygons
    - tolerance (float): Tolerance for simplification (in CRS units)

    Returns:
    - GeoDataFrame: Simplified polygons
    """
    gdf = gdf.copy()
    gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
    return gdf

#=====================================================#
# Attribute Table Operations
#=====================================================#

def list_zonal_stats(polygons, param_list, crs, transform):
    """
    Calculate zonal statistics for a list of polygons and parameters.
    
    Parameters:
    - polygons (list): List of polygon geometries.
    - param_list (list): List of parameters for each polygon.
    - crs: Coordinate reference system.
    - transform: Affine transform for the raster.
    
    Returns:
    - list: Zonal statistics for each polygon.
    """
    results = []

    x_res = transform[0]
    y_res = abs(transform[4])  # y res is negative for north-up images
    pixel_area = x_res * y_res

    results = gpd.GeoDataFrame()
    for param in param_list:
        temp = zonal_stats(polygons, param.raster, pixel_area, crs, transform)
        results = pd.concat([results, temp], ignore_index=True)
    return results

def zonal_stats(vector_layers, data_raster, pixel_area, crs, transform):
    param_name = "D2300"

    stats = gpd.GeoDataFrame()
    base_raster = array_to_rasterio(data_raster, transform, crs)

    vector_stack = merge_polygons(vector_layers[1:]) # Skip the first layer as it is the mask
    # vector_stack = merge_polygons(vector_layers)

    temp = exact_extract(
        base_raster,
        vector_stack,
        [
            f"{param_name}_mean=mean",
            f"{param_name}_median=median",
            f"{param_name}area=count",
            f"{param_name}_min=min",
            f"{param_name}_max=max",
            f"{param_name}_p=quantile(q=0.25)",
            f"{param_name}_p=quantile(q=0.75)",
            f"{param_name}_sd=stdev"
        ],
        include_geom=True,
        include_cols="value",
        strategy="raster-sequential", # works when rasters are simplified 
        output='pandas',
        progress=True
    )

    stats = pd.concat([stats, temp], ignore_index=True)
    
    stats[f"{param_name}area"] = stats[f"{param_name}area"] * pixel_area * 0.001  # Convert to square kilometers
 
    return stats

def array_to_rasterio(array, transform, crs):
    height, width = array.shape
    memfile = MemoryFile()
    with memfile.open(
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=array.dtype,
        transform=transform,
        crs=crs
    ) as dataset:
        dataset.write(array, 1)
    return memfile.open()