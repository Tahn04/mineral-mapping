import os
import gc
import rasterio as rio
from rasterio import features
from rasterio.io import MemoryFile
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
from exactextract import exact_extract
from tqdm import tqdm

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

def list_sieve_filter(array, iterations=1, threshold=9, connectedness=4, crs=None, transform=None):
    array = np.asarray(array)
    bands, height, width = array.shape
    filtered_array = np.empty_like(array, dtype="uint8")

    crs_wkt = crs.to_wkt()

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

def vectorize(mask_array, value, transform, crs):
    # Convert boolean mask to uint8 for rasterio.features.shapes
    mask_uint8 = mask_array.astype(np.uint8)
    shapes = features.shapes(mask_uint8, transform=transform) # should be rio.features.shapes
    polygons = [
        {"geometry": shape(geom), "value": value}
        for geom, val in shapes if val == 1
    ]
    gdf = gpd.GeoDataFrame(polygons, crs=crs)

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
    # sd = ndimage.standard_deviation(data_clean, labels=zones_clean, index=unique_zones)
    counts = ndimage.sum(np.ones_like(zones_clean), labels=zones_clean, index=unique_zones)
    minimum = ndimage.minimum(data_clean, labels=zones_clean, index=unique_zones)
    # maximum = ndimage.maximum(data_clean, labels=zones_clean, index=unique_zones)
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
            # 'std': round(float(s), 6),
            'min': round(float(mi), 6),
            # 'max': round(float(ma), 6),
            'p25': round(percentiles[int(zone)]['p25'], 6),
            'p75': round(percentiles[int(zone)]['p75'], 6),
            'area': round(float(count * pixel_area), 3),
        }
        for mi, zone, mean, count in zip(minimum, unique_zones, means, counts)
    }

def zonal_stats3(vector_layers, data_raster, pixel_area, crs, transform):
    """
    Calculate zonal statistics for a given data raster and polygon.
    
    Parameters:
    - vector_layers (list of gpd.GeoDataFrame): List of GeoDataFrames containing polygons for each layer.
    - data_raster (np.ndarray): 2D array of data values.
    - value (float): The value of the raster as a whole (threshold/confidence level).
    - pixel_area (float): Area represented by each pixel.
    
    Returns:
    - dict: Zonal statistics for the polygon.
    """
    param_name = "D2300"
    stats = gpd.GeoDataFrame()
    base_raster = array_to_rasterio(data_raster, transform, crs)
    for vector_layer in tqdm(vector_layers, desc="Calculating zonal stats"):
        n_chunks = len(polygon)
        polygon_chunks = np.array_split(polygon, n_chunks)
        intersting_chuck = polygon_chunks[144] 
        # show_polygons(intersting_chuck, title=f"Chunk")
        stats_list = []
        for i, poly_chunk in enumerate(polygon_chunks):
            print(f"Processing chunk {i + 1}/{n_chunks}")
            # show_polygons(poly_chunk, title=f"Chunk {i+1}")
            if i != 144:
                temp = exact_extract(
                    base_raster,
                    poly_chunk,
                    [
                        f"{param_name}_mean=mean",
                        f"{param_name}area=count",
                        f"{param_name}_min=min",
                        f"{param_name}_p25=quantile(q=0.25)",
                        f"{param_name}_p75=quantile(q=0.75)",
                        f"{param_name}_sd=stdev"
                    ],
                    include_geom=True,
                    include_cols="value",
                    output='pandas',
                    progress=True
                )
                stats_list.append(temp)
            else:
                show_polygons(poly_chunk, title=f"Chunk")
                temp = exact_extract(
                    base_raster,
                    poly_chunk,
                    [
                        f"{param_name}_mean=mean",
                        f"{param_name}area=count",
                        f"{param_name}_min=min",
                        f"{param_name}_p25=quantile(q=0.25)",
                        f"{param_name}_p75=quantile(q=0.75)",
                        f"{param_name}_sd=stdev"
                    ],
                    include_geom=True,
                    strategy="raster-sequential",
                    include_cols="value",
                    output='pandas',
                    progress=True
                )
                stats_list.append(temp)

        stats = pd.concat(stats_list, ignore_index=True)
    
    stats[f"{param_name}area"] = stats[f"{param_name}area"] * pixel_area * 1000000  # Convert to square meters
 
    return stats

def zonal_stats2(vector_layers, data_raster, pixel_area, crs, transform):
    param_name = "D2300"

    stats = gpd.GeoDataFrame()
    base_raster = array_to_rasterio(data_raster, transform, crs)

    vector_stack = merge_polygons(vector_layers[1:])

    temp = exact_extract(
        base_raster,
        vector_stack,
        [
            f"{param_name}_mean=mean",
            f"{param_name}area=count",
            f"{param_name}_min=min",
            f"{param_name}_p25=quantile(q=0.25)",
            f"{param_name}_p75=quantile(q=0.75)",
            f"{param_name}_sd=stdev"
        ],
        include_geom=True,
        include_cols="value",
        strategy="raster-sequential",
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