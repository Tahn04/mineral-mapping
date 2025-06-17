import os
import gc
import rasterio as rio
from rasterio import features
from rasterio.io import MemoryFile
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import convolve2d
from scipy.ndimage import convolve
from shapely.geometry import shape
from skimage.morphology import dilation, erosion, square
from osgeo import gdal
import rasterio
import tempfile
from scipy import ndimage
from exactextract import exact_extract
from tqdm import tqdm
import dask.array as da
from affine import Affine

#=====================================================#
# Raster Operations
#=====================================================#

def show_raster(raster, cmap='gray', title=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    img = plt.imshow(raster, cmap=cmap)
    plt.colorbar(img, label='Value')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def threshold(raster, threshold):
    raster = np.asarray(raster)
    return (raster > threshold).astype(raster.dtype)

def mask(raster, mask):
    return raster * mask

def open_raster(raster_path):
    return rio.open(raster_path) 

def open_raster_band(raster, band_number):
    return raster.read(band_number, masked=True).filled(np.nan)

def save_raster(raster, output_path, file_name, profile):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    full_path = os.path.join(output_path, file_name)
    raster = np.asarray(raster)
    with rio.open(full_path, 'w', **profile) as dst:
        # If raster is 2D, add a band axis
        if raster.ndim == 2:
            dst.write(raster, 1)
        else:
            dst.write(raster)

#=====================================================#
# Processing Functions
#=====================================================#

def majority_filter(binary_array, size=3, iterations=1):
    kernel = np.ones((size, size), dtype=np.uint8)

    binary_array = np.nan_to_num(binary_array, nan=0).astype(np.uint8)

    for _ in range(iterations):
        count = convolve2d(binary_array, kernel, mode='same', boundary='symm')
        threshold = (size * size) // 2
        binary_array = (count > threshold).astype(np.uint8)

    return binary_array

def majority_filter_fast(binary_array, size=3, iterations=1):
    kernel = np.ones((size, size), dtype=np.uint8)
    array = np.nan_to_num(binary_array, nan=0).astype(np.uint8)
    threshold = (size * size) // 2

    for _ in range(iterations):
        count = convolve(array, kernel, mode='mirror')
        array = (count > threshold).astype(np.uint8)

    return array

def dask_majority_filter(arr, size=3, iterations=1):
    def majority_func(block):
        return majority_filter_fast(block, size=size, iterations=iterations)
    
    dask_arr = da.from_array(arr, chunks=(1024, 1024))
    depth = size // 2

    return dask_arr.map_overlap(majority_func, depth=depth, boundary=0).compute()

def boundary_clean(raster_array, classes=None, iterations=2, radius=3):

    if classes is None:
        classes = np.unique(raster_array[~np.isnan(raster_array)])

    result = np.copy(raster_array)
    selem = square(radius)

    for _ in range(iterations):
        for cls in classes:
            mask = result == cls
            closed = erosion(dilation(mask, selem), selem)
            result[closed] = cls 

    return result

def list_sieve_filter(array, crs, transform, iterations=1, threshold=9, connectedness=4):
    array = np.asarray(array)
    bands, height, width = array.shape
    filtered_array = np.empty_like(array, dtype="uint8")

    crs_wkt = crs.to_wkt()
    # crs_wkt = crs

    for b in tqdm(range(bands), desc="Applying Sieve Filter"):
        array_uint8 = np.nan_to_num(array[b], nan=0).astype("uint8")

        # Initialize source in-memory dataset
        src_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
        src_ds.SetGeoTransform(transform.to_gdal())  # rasterio transform -> GDAL format
        src_ds.SetProjection(crs_wkt)
        src_ds.GetRasterBand(1).WriteArray(array_uint8)

        for _ in range(iterations):
            # Create new MEM dataset for output
            dst_ds = gdal.GetDriverByName("MEM").Create("", width, height, 1, gdal.GDT_Byte)
            dst_ds.SetGeoTransform(transform.to_gdal())
            dst_ds.SetProjection(crs_wkt)

            # Apply sieve filter
            gdal.SieveFilter(
                srcBand=src_ds.GetRasterBand(1),
                maskBand=None,
                dstBand=dst_ds.GetRasterBand(1),
                threshold=threshold,
                connectedness=connectedness
            )

            # Swap for next iteration
            src_ds = dst_ds

        # Read back result
        filtered_array[b] = dst_ds.GetRasterBand(1).ReadAsArray()

    return filtered_array

#=====================================================#
# Vector Operations
#=====================================================#

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

def vectorize_dict(labeled_array, stats_dict, transform, crs):
    features = rio.features.shapes(labeled_array.astype(np.int32), transform=transform)

    polygons = []
    for geom, label in features:
        label = int(label)
        if label == 0 or label not in stats_dict:
            continue

        data = {"geometry": shape(geom), "label": label}
        data.update(stats_dict[label])
        polygons.append(data)

    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    return gdf

def merge_polygons(gdf):
    return pd.concat(gdf, ignore_index=True)
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

def label_clusters(binary_raster, connectivity=1):
    """
    Labels connected regions of 1s in a binary raster.
    
    Parameters:
    - binary_raster (np.ndarray): 2D array of 0s and 1s.
    - connectivity (int): 1 for 4-connected, 2 for 8-connected (diagonals included).
    
    Returns:
    - labeled (np.ndarray): Same shape as input, with unique labels for each cluster.
    """
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled = ndimage.label(binary_raster, structure=structure)[0]
    return labeled

def zonal_stats2(vector_layers, data_raster, pixel_area, crs, transform):
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
    
    stats[f"{param_name}area"] = stats[f"{param_name}area"] * pixel_area * 0.001  # Convert to square meters
 
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