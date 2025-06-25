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
from exactextract.writer import GDALWriter
from rasterio.features import shapes
import dask.array as da
from osgeo import gdal, osr, ogr
import tempfile



def list_vectorize(raster_list, thresholds, crs, transform, simplify_tol=200):
    """
    Vectorizes a list of rasters using corresponding threshold values.

    Parameters:
    - raster_list (list of np.ndarray): List of binary rasters.
    - thresholds (list of float or int): Threshold values associated with each raster.
    - crs: Coordinate Reference System (e.g., from rasterio).
    - transform: Affine transform (e.g., from rasterio).
    - simplify_tol: Simplification tolerance in map units.

    Returns:
    - List of GeoDataFrames
    """
    results = [
        # vectorize(raster, threshold, transform, crs, simplify_tol=simplify_tol)
        dask_vectorize(raster, transform, crs, simplify_tol=simplify_tol)
        for raster, threshold in tqdm(zip(raster_list, thresholds), desc="Vectorizing", total=len(raster_list))
    ]
    # show_polygons(results[1], title="Vectorized Raster")
    return results

def vectorize_chunk(chunk, transform, value=1, simplify_tol=0):
    """
    Vectorize a chunk (NumPy array).
    Return a list of GeoJSON-like dicts.
    """
    result = []
    transform = Affine(*transform)  # Ensure it's an Affine

    for geom, val in shapes(chunk.astype("int32"), transform=transform):
        if val == value:
            poly = shape(geom)
            if simplify_tol:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            result.append({"geometry": poly, "value": value})
    return result

def dask_vectorize(array, transform, crs, chunk_size=(512, 512), value=1, simplify_tol=0):
    """
    Vectorize a large raster using Dask with blockwise vectorization. (256, 256) (512, 512)

    Parameters:
    - array: 2D NumPy array or Dask array
    - transform: Affine transform (rasterio-style)
    - crs: CRS (string, dict, or pyproj.CRS)
    - chunk_size: size of chunks to break the array into
    - value: pixel value to vectorize
    - simplify_tol: simplification tolerance

    Returns:
    - GeoDataFrame with vectorized polygons
    """
    if not isinstance(array, da.Array):
        array = da.from_array(array, chunks=chunk_size)

    results = []
    affine_transform = Affine(transform[1], transform[2], transform[0],
                               transform[4], transform[5], transform[3])
    for i in range(0, array.shape[0], chunk_size[0]):
        for j in range(0, array.shape[1], chunk_size[1]):
            block = array[i:i+chunk_size[0], j:j+chunk_size[1]].compute()
            if np.any(block == value):
                block_transform = affine_transform * Affine.translation(j, i)
                geoms = vectorize_chunk(block, block_transform, value, simplify_tol, threshold=threshold)

                results.extend(geoms)

    return gpd.GeoDataFrame(results, crs=crs)

def merge_polygons(gdfs):
    """
    Merge a list of GeoDataFrames into a single GeoDataFrame.
    If threshold values are present, dissolves polygons with the same threshold
    to handle polygons that were split across tile boundaries.
    
    Parameters:
    - gdfs: List of GeoDataFrames from tiled vectorization
    
    Returns:
    - GeoDataFrame with properly merged polygons
    """
    if not gdfs:
        return gpd.GeoDataFrame()
    
    # Concatenate all GeoDataFrames
    merged = pd.concat(gdfs, ignore_index=True)
    
    # # If we have threshold values, dissolve by threshold to merge split polygons
    # if 'threshold' in merged.columns:
    #     # Dissolve polygons with the same threshold value
    #     dissolved = merged.dissolve(by='threshold', as_index=True)
    #     # Convert back to a regular GeoDataFrame and reset index
    #     return dissolved.reset_index()
    
    return merged

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

def save_tiled_raster(input_array, transform, crs, output_path, tile_size=512):
    driver = gdal.GetDriverByName("GTiff")
    height, width = input_array.shape

    dst_ds = driver.Create(
        output_path,
        width,
        height,
        1,
        gdal.GDT_Float32,
        options=[
            "TILED=YES",
            f"BLOCKXSIZE={tile_size}",
            f"BLOCKYSIZE={tile_size}",
            "COMPRESS=DEFLATE"
        ]
    )
    dst_ds.SetGeoTransform(transform)
    dst_ds.SetProjection(crs)
    dst_ds.GetRasterBand(1).WriteArray(input_array)
    dst_ds.FlushCache()
    return dst_ds


def get_tiled_raster_path(param):
    if not hasattr(param, '_tiled_path'):
        temp_dir = tempfile.mkdtemp()
        param._tiled_path = os.path.join(temp_dir, f"{param.name}_tiled.tif")
        save_tiled_raster(param.raster, param.get_transform(), param.get_crs(), param._tiled_path)
    return param._tiled_path


def list_zonal_stats(polygons, param_list, crs, transform):
    results = gpd.GeoDataFrame()

    x_res = transform[1]
    y_res = abs(transform[5])
    pixel_area = x_res * y_res

    for param in param_list:
        temp = zonal_stats(polygons, param, pixel_area)

        if results.empty:
            results = temp
        else:
            results = results.join(temp.set_index(results.index), rsuffix=f"_{param.name}")
            if f"geometry_{param.name}" in results.columns:
                results = results.drop(columns=[f"geometry_{param.name}"])
    return results


def zonal_stats(vector_layers, param, pixel_area):
    stats = gpd.GeoDataFrame()
    raster_path = get_tiled_raster_path(param)
    vector_stack = merge_polygons(vector_layers)
    import matplotlib.pyplot as plt

    with rasterio.open(raster_path) as src:
        img = src.read(1)
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap='viridis')
        plt.title(f"Raster: {os.path.basename(raster_path)}")
        plt.colorbar(label='Value')
        plt.axis('off')
        plt.show()

    operations = [
        f"{param.name}_M=mean",
        f"{param.name}_MDN=median",
        f"{param.name}_SQKM=count",
        f"{param.name}_MIN=min",
        f"{param.name}_MAX=max",
        f"{param.name}=quantile(q=0.25)",
        f"{param.name}=quantile(q=0.75)",
        f"{param.name}_SD=stdev"
    ]

    temp = exact_extract(
        raster_path,
        vector_stack,
        operations,
        include_geom=True,
        include_cols="threshold",
        # strategy="raster-sequential",
        output='pandas',
        progress=True
    )

    stats = pd.concat([stats, temp], ignore_index=True)
    stats[f"{param.name}_SQKM"] = stats[f"{param.name}_SQKM"] * pixel_area * 0.000001

    float_cols = stats.select_dtypes(include=['float']).columns
    stats[float_cols] = stats[float_cols].round(4)
    return stats
# def list_zonal_stats(polygons, param_list, crs, transform):
#     """
#     Calculate zonal statistics for a list of polygons and parameters.
    
#     Parameters:
#     - polygons (list): List of polygon geometries.
#     - param_list (list): List of parameters for each polygon.
#     - crs: Coordinate reference system.
#     - transform: Affine transform for the raster.
    
#     Returns:
#     - list: Zonal statistics for each polygon.
#     """
#     results = []

#     x_res = transform[0]
#     y_res = abs(transform[4])  # y res is negative for north-up images
#     pixel_area = x_res * y_res

#     results = gpd.GeoDataFrame()
#     for param in param_list:
#         temp = zonal_stats(polygons, param.raster, param.dataset, pixel_area, crs, transform, param.name)
#         if results.empty:
#             results = temp
#         else:
#             results = results.join(temp.set_index(results.index), rsuffix=f"_{param.name}")
#             if f"geometry_{param.name}" in results.columns:
#                 results = results.drop(columns=[f"geometry_{param.name}"])
#                 # results = results.drop(columns=[f"value_{param.name}"])
#     return results

# def zonal_stats(vector_layers, data_raster, dataset, pixel_area, crs, transform, param_name):
#     """ Calculate zonal statistics for a raster and vector layers."""
#     stats = gpd.GeoDataFrame()
#     if dataset is not None:
#         base_raster = dataset
#     else:
#         base_raster = array_to_gdal(data_raster, transform, crs)
    
#     # vector_stack = merge_polygons(vector_layers[1:]) # Skip the first layer as it is the mask
#     vector_stack = merge_polygons(vector_layers)

#     temp = exact_extract(
#         base_raster,
#         vector_stack,
#         [
#             f"{param_name}_M=mean",
#             f"{param_name}_MDN=median",
#             f"{param_name}_SQKM=count",
#             f"{param_name}_MIN=min",
#             f"{param_name}_MAX=max",
#             f"{param_name}=quantile(q=0.25)",
#             f"{param_name}=quantile(q=0.75)",
#             f"{param_name}_SD=stdev"
#         ],
#         include_geom=True,
#         include_cols="threshold",
#         strategy="raster-sequential", # works when rasters are simplified 
#         output='pandas',
#     #     output_options={
#     #     "filename": "zonal_stats.shp",
#     #     "driver": "ESRI Shapefile"
#     # },
#         progress=True
#     )

#     stats = pd.concat([stats, temp], ignore_index=True)

#     stats[f"{param_name}_SQKM"] = stats[f"{param_name}_SQKM"] * pixel_area * 0.000001  # Convert to square kilometers
  
#     float_cols = stats.select_dtypes(include=['float']).columns
#     stats[float_cols] = stats[float_cols].round(4) 
#     return stats

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

def array_to_gdal(array, transform, crs):
    """ Convert a NumPy array to an in-memory GDAL raster dataset. """
    height, width = array.shape
    mem_driver = gdal.GetDriverByName('MEM')
    dataset = mem_driver.Create('', width, height, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(transform)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs)
    dataset.SetProjection(srs.ExportToWkt())

    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    band.FlushCache()
    return dataset

