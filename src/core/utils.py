import os

import rasterio as rio
from rasterio import features
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.signal import convolve2d
from shapely.geometry import shape
from skimage.morphology import dilation, erosion, square
from osgeo import gdal
import rasterio
import tempfile
from scipy import ndimage

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

def list_sieve_filter(array, threshold=9, connectedness=4, profile=None, iterations=1):
    array = np.asarray(array)
    bands, height, width = array.shape
    filtered_array = np.empty_like(array, dtype="uint8")

    transform = profile.get_transform()
    crs = profile.get_crs()

    for b in range(bands):
        array_uint8 = np.nan_to_num(array[b], nan=0).astype("uint8")

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp2:

            tmp_src_path, tmp_dst_path = tmp1.name, tmp2.name

            # Write band to file
            with rio.open(
                tmp_src_path, "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="uint8",
                crs=crs,
                transform=transform,
            ) as ds:
                ds.write(array_uint8, 1)

            # Apply sieve filtering iteratively
            for i in range(iterations):
                src_path = tmp_src_path if i % 2 == 0 else tmp_dst_path
                dst_path = tmp_dst_path if i % 2 == 0 else tmp_src_path

                src_ds = gdal.Open(src_path, gdal.GA_ReadOnly)
                dst_ds = gdal.GetDriverByName("GTiff").Create(
                    dst_path,
                    width,
                    height,
                    1,
                    gdal.GDT_Byte
                )

                gdal.SieveFilter(
                    srcBand=src_ds.GetRasterBand(1),
                    maskBand=None,
                    dstBand=dst_ds.GetRasterBand(1),
                    threshold=threshold,
                    connectedness=connectedness
                )

                src_ds = None
                dst_ds = None

            # Read final result
            final_ds = gdal.Open(dst_path, gdal.GA_ReadOnly)
            filtered_array[b] = final_ds.GetRasterBand(1).ReadAsArray()
            final_ds = None

            # # Clean up temp files
            # os.unlink(tmp_src_path)
            # os.unlink(tmp_dst_path)

    return filtered_array

#=====================================================#
# Vector Operations
#=====================================================#

def vectorize(mask_array, value, transform, crs):
    # Convert boolean mask to uint8 for rasterio.features.shapes
    mask_uint8 = mask_array.astype(np.uint8)
    shapes = features.shapes(mask_uint8, transform=transform) # should be rio.features.shapes
    polygons = [
        {"geometry": shape(geom), "value": value}
        for geom, val in shapes if val == 1
    ]
    gdf = gpd.GeoDataFrame(polygons, crs=crs)
    # gdf = gdf.drop(columns="value")
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

def save_shapefile(gdf, output_path, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gdf.to_file(os.path.join(output_path, file_name), driver='ESRI Shapefile')

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

def zonal_stats(zone_raster, data_raster, value, pixel_area):
    """
    Calculate zonal statistics for a given data raster and labeled raster.
    Parameters:
    - data_raster (np.ndarray): 2D array of data values.
    - zone_raster (np.ndarray): 2D array of zone labels.
    - value (float): The value of the raster as a whole (threshold/confidence level).
    - pixel_area (float): Area represented by each pixel 
    """
    data_mask = ~np.isnan(data_raster)
    data_clean = data_raster[data_mask]
    zones_clean = zone_raster[data_mask]

    unique_zones = np.unique(zones_clean)

    means = ndimage.mean(data_clean, labels=zones_clean, index=unique_zones)
    sd = ndimage.standard_deviation(data_clean, labels=zones_clean, index=unique_zones)
    minimum = ndimage.minimum(data_clean, labels=zones_clean, index=unique_zones)
    maximum = ndimage.maximum(data_clean, labels=zones_clean, index=unique_zones)
    counts = ndimage.sum(np.ones_like(zones_clean), labels=zones_clean, index=unique_zones)
    percentiles = {
        int(zone): {
            'p25': float(np.percentile(data_clean[zones_clean == zone], 25)),
            'p75': float(np.percentile(data_clean[zones_clean == zone], 75))
        }
        for zone in unique_zones
    }

    # Combine stats into final dictionary
    return {
        int(zone): {
            'value': float(value),
            'mean': round(float(mean), 6),
            'std': round(float(s), 6),
            'min': round(float(mi), 6),
            'max': round(float(ma), 6),
            'p25': round(percentiles[int(zone)]['p25'], 6),
            'p75': round(percentiles[int(zone)]['p75'], 6),
            'area': round(float(count * pixel_area), 3),
        }
        for zone, mean, s, mi, ma, count in zip(unique_zones, means, sd, minimum, maximum, counts)
    }