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
from rasterio.features import shapes
import dask.array as da
from osgeo import gdal, osr, ogr
import tempfile
from tempfile import TemporaryDirectory
import rasterio
from rasterio.mask import mask
from scipy import ndimage
import core.raster_ops as ro
import core.file_handler as fh
import bottleneck as bn
# import fiona
import gc

def list_raster_to_shape_gdal(raster_list, thresholds, crs, transform, param_list, stats_list, simplification_level=0):
    file_paths = []
    for raster, threshold in tqdm(zip(raster_list, thresholds), desc="Converting rasters to shapes"):
        vector_file = fh.FileHandler().create_temp_file(prefix=f"{threshold}_shapes", suffix='shp')
        raster_to_shape_gdal(raster.astype(np.uint8), transform, crs, vector_file, threshold=threshold)
        file_paths.append(vector_file)

    gdf = list_file_zonal_stats(file_paths, param_list, crs, transform, stats_list, simplification_level)

    return gdf

def raster_to_shape_gdal(binary_array, transform, crs_wkt, vector_file, threshold=0):
    """
    Convert a binary numpy array to polygons and save as a shapefile.

    Parameters:
    - binary_array: 2D numpy array (binary mask)
    - transform: affine.Affine transform for the raster
    - crs_wkt: CRS in WKT format
    - vector_file: output shapefile path
    """

    # Create an in-memory raster from the array
    driver = gdal.GetDriverByName('MEM')
    rows, cols = binary_array.shape
    mem_raster = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(transform)
    mem_raster.SetProjection(crs_wkt)
    mem_raster.GetRasterBand(1).WriteArray(binary_array.astype(np.uint8))

    # Prepare shapefile
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = shp_driver.CreateDataSource(vector_file)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    layer = shp_ds.CreateLayer('layername', srs=srs)
    field = ogr.FieldDefn('ID', ogr.OFTInteger)
    layer.CreateField(field)

    threshold_field = ogr.FieldDefn('Threshold', ogr.OFTReal)
    layer.CreateField(threshold_field)

    # Polygonize
    # Only polygonize where the raster is 1 (not 0)
    mask_band = mem_raster.GetRasterBand(1)
    gdal.Polygonize(mask_band, mask_band, layer, 0, [], callback=None)

    layer.ResetReading()
    for feature in layer:
        feature.SetField('Threshold', float(threshold))
        layer.SetFeature(feature)

    layer = None 
    shp_ds.Destroy() 
    shp_ds = None
    mem_raster = None
    mask_band = None

def list_file_zonal_stats(path_list, param_list, crs, transform, stats_list, simplification_level=0):
    """
    Compute zonal statistics for a list of raster files and a list of parameters.

    Args:
        path_list (list): List of file paths to the raster files.
        param_list (list): List of parameters to compute statistics for.
        crs: Coordinate Reference System (e.g., from rasterio).
        transform: Affine transform (e.g., from rasterio).
        stats_list (list): List to store the computed statistics.

    Returns:
        GeoDataFrame with zonal statistics.
    """
    results = gpd.GeoDataFrame()
    x_res = transform[1]
    y_res = abs(transform[5])  # y res is negative for north-up images
    pixel_area = x_res * y_res
    for param in param_list:
        stats_config = config_stats(stats_list, param.name)
        temp = file_zonal_stats(path_list, param, crs, transform, stats_config, pixel_area, simplification_level)
        if results.empty:
            results = temp
        else:
            results = results.join(temp.set_index(results.index), rsuffix=f"_{param.name}")
            if f"geometry_{param.name}" in results.columns:
                results = results.drop(columns=[f"geometry_{param.name}"])
                results = results.drop(columns=[f"Threshold_{param.name}"])

    return results

def file_zonal_stats(path_list, param, crs, transform, stats_config, pixel_area, simplification_level):
    """
    Compute zonal statistics for a list of raster files and a list of parameters.

    Parameters:
    - path_list: List of file paths to the raster files.
    - param_list: List of parameters to compute statistics for.
    - crs: Coordinate Reference System (e.g., from rasterio).
    - transform: Affine transform (e.g., from rasterio).
    - stats_list: List to store the computed statistics.

    Returns:
    - GeoDataFrame with zonal statistics.
    """
    gdf = gpd.GeoDataFrame()
    param_name = param.name
    if len(stats_config) != 0:
        if param.raster_path is not None:
                base_raster = param.raster_path
        else:
            base_raster = array_to_gdal(param.raster, transform, crs)
        # param.release()

    with gdal.Open(base_raster) as rast:
        # for path in tqdm(path_list, desc=f"Calculating zonal stats for {param.name}"):
        #     temp = gpd.read_file(path)
        #     gdf = pd.concat([gdf, temp], ignore_index=True)
        # if len(stats_config) != 0:
        #     # with ogr.Open(path) as vect:
        #     # pre_gdf = gpd.read_file(path)
        #     # pre_gdf = pre_gdf.dissolve(by='Threshold', as_index=False)  

        #     temp = exact_extract(
        #         rast,
        #         gdf,
        #         stats_config,
        #         include_geom=True,
        #         include_cols="Threshold",
        #         # strategy="raster-sequential",
        #         output='pandas',
        #         progress=True,
        #         max_cells_in_memory=1000000000
        #     )
        #     gdf = gpd.GeoDataFrame()
        #     gdf = pd.concat([gdf, temp], ignore_index=True)

        #     if f"{param_name}_SQK" in gdf.columns:
        #         gdf[f"{param_name}_SQK"] = gdf[f"{param_name}_SQK"] * pixel_area * 0.000001  # Convert to square kilometers

        #     float_cols = gdf.select_dtypes(include=['float']).columns
        #     gdf[float_cols] = gdf[float_cols].round(4) 
        # else:
        #     temp = gpd.read_file(path)
        #     gdf = pd.concat([gdf, temp], ignore_index=True)
        for path in tqdm(path_list, desc=f"Calculating zonal stats for {param.name}"):
            if len(stats_config) != 0:
                with ogr.Open(path) as vect:
                # pre_gdf = gpd.read_file(path)
                # pre_gdf = pre_gdf.dissolve(by='Threshold', as_index=False)  
                    temp = exact_extract(
                        rast,
                        vect,
                        stats_config,
                        include_geom=True,
                        include_cols="Threshold",
                        # strategy="raster-sequential",
                        output='pandas',
                        progress=True,
                        max_cells_in_memory=10000000000
                    )

                gdf = pd.concat([gdf, temp], ignore_index=True)

                if f"{param_name}_SQK" in gdf.columns:
                    gdf[f"{param_name}_SQK"] = gdf[f"{param_name}_SQK"] * pixel_area * 0.000001  # Convert to square kilometers

                float_cols = gdf.select_dtypes(include=['float']).columns
                gdf[float_cols] = gdf[float_cols].round(4) 
            else:
                temp = gpd.read_file(path)
                gdf = pd.concat([gdf, temp], ignore_index=True)

    return gdf

def list_vectorize(raster_list, thresholds, crs, transform, simplify_tol):
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
        dask_vectorize(raster, transform, crs, threshold=threshold, simplify_tol=simplify_tol)
        for raster, threshold in tqdm(zip(raster_list, thresholds), desc="Vectorizing", total=len(raster_list))
    ]
    results = combine_polygons(results)
    return results

def vectorize_chunk(chunk, transform, value=1, simplify_tol=0, threshold=None):
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
            feature = {"geometry": poly}
            if threshold is not None:
                feature["Threshold"] = threshold
            result.append(feature)
    return result

def dask_vectorize(array, transform, crs, chunk_size=(512, 512), value=1, simplify_tol=0, threshold=None):
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
                geoms = vectorize_chunk(block, block_transform, value, simplify_tol, threshold)
                results.extend(geoms)
    if results:
        return gpd.GeoDataFrame(results, crs=crs)
    else:
        return gpd.GeoDataFrame()


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
    file_path = os.path.join(output_path, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    gdf.to_file(file_path, driver=driver)

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

def save_tiled_raster(input_array, transform, crs, output_path, tile_size=256):
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

def combine_polygons(gdf):
    """
    Combine polygons from a list of GeoDataFrames by merging geometries that touch or overlap.
    This helps to reconstruct polygons that were split during tiling.

    Parameters:
    - gdf_list: List of GeoDataFrames

    Returns:
    - GeoDataFrame with merged polygons
    """
    # merged = pd.concat(gdf_list[1:], ignore_index=True)
    if isinstance(gdf, list):
        gdf = pd.concat(gdf, ignore_index=True)
    dissolved = gdf.dissolve(by='Threshold', as_index=False)
    separated = dissolved.explode(index_parts=True)
    # separated = pd.concat([gdf_list[0], separated], ignore_index=True) # Add the first GeoDataFrame (mask) back to the merged result
    cleaned = separated.reset_index(drop=True)
    return cleaned

def list_zonal_stats(polygons, param_list, crs, transform, stats_list):
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

    x_res = transform[1]
    y_res = abs(transform[5])  # y res is negative for north-up images
    pixel_area = x_res * y_res
    gdf = combine_polygons(polygons[1:])
    # gdf = polygons[0:2]
    results = gpd.GeoDataFrame()
    for param in param_list:
        stats_config = config_stats(stats_list, param.name)  # Get the configured stats for the parameter
        
        # raster_path = get_tiled_raster_path(param)
        temp = zonal_stats(gdf, param.raster, param.dataset, pixel_area, crs, transform, param.name, stats_config, param)
        # temp = tiled_zonal_stats(gdf, raster_path, stats_config, tile_size=2048, overlap=100, temp_dir=None, cleanup=True, strategy="raster-sequential")
        if results.empty:
            results = temp
        else:
            results = results.join(temp.set_index(results.index), rsuffix=f"_{param.name}")
            if f"geometry_{param.name}" in results.columns:
                results = results.drop(columns=[f"geometry_{param.name}"])
                # results = results.drop(columns=[f"value_{param.name}"])
    return results

def zonal_stats(gdf, data_raster, dataset, pixel_area, crs, transform, param_name, stats_config, param):
    """ Calculate zonal statistics for a raster and vector layers."""
    if len(stats_config) != 0:
        empty_gdf = gpd.GeoDataFrame()
        # if dataset is not None:
        #     base_raster = dataset
        # else:
        #     base_raster = array_to_gdal(data_raster, transform, crs)
        base_raster = array_to_gdal(data_raster, transform, crs)
        raster_path = get_tiled_raster_path(param)
        param.release()  # Release the raster dataset to avoid memory issues
        # temp = rioxarray_zonal_stats(gdf, raster_path, stat="median")
        temp = exact_extract(
            raster_path,
            gdf,
            stats_config,
            include_geom=True,
            include_cols="Threshold",
            # strategy="raster-sequential",
            output='pandas',
            progress=True,
            max_cells_in_memory=1000000000 # Adjust as needed for large datasets
        )
        # temp_dir = os.path.dirname(raster_path)
        # shutil.rmtree(temp_dir)
        gdf = pd.concat([empty_gdf, temp], ignore_index=True)

        gdf[f"{param_name}_SQK"] = gdf[f"{param_name}_SQK"] * pixel_area * 0.000001  # Convert to square kilometers
    
        float_cols = gdf.select_dtypes(include=['float']).columns
        gdf[float_cols] = gdf[float_cols].round(4) 
    return gdf

def config_stats(stats_list, param_name):
    """configure statistics for a list of stats."""
    stat_config = []
    stats_map = {
            'mean': f"{param_name}_MEN=mean",
            'median': f"{param_name}_MDN=median",
            'area': f"{param_name}_SQK=count",
            'count': f"{param_name}_CNT=count",
            'min': f"{param_name}_MIN=min",
            'max': f"{param_name}_MAX=max",
            'std': f"{param_name}_STD=stdev",
        }
    # stats_map = {
    #         'mean': "MEN=mean",
    #         'median': "MDN=median",
    #         'area': "SQK=count",
    #         'count': "CNT=count",
    #         'min': "MIN=min",
    #         'max': "MAX=max",
    #         'std': "STD=stdev",
    #     }
    for stat in stats_list:
        if isinstance(stat, str) and stat.endswith('p'):
            if len(stat) < 2 or not stat[:-1].isdigit():
                raise ValueError(f"Invalid percentile format: {stat}. Must be a number followed by 'p'.")
            p = float(stat[:-1])
            stat_config.append(f"Q=quantile(q={p/100})")
        elif stat in stats_map:
            stat_config.append(stats_map[stat])
        else:
            raise ValueError(f"Statistic '{stat}' is not supported. Supported statistics are: {list(stats_map.keys())}")
    return stat_config

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

def list_raster_stats(param_list, raster_list, stats, thresholds):
    for param in param_list:
        base_raster = param.get_raster()
        for raster, threshold in zip(raster_list, thresholds):
            # print(f"Calculating zonal stats for {param.name} with threshold {threshold}")
            labeled_raster = ro.label_clusters(raster)
            results = scipy_zonal_stats(base_raster, labeled_raster, stats)

def scipy_zonal_stats(base_raster, labeled_raster, stats):
    """
    Calculate zonal statistics using SciPy for a GeoDataFrame and a raster file.
    
    Parameters:
    - gdf (GeoDataFrame): GeoDataFrame with polygon geometries.
    - raster_path (str): Path to the raster file.
    - stat (str): Statistic to calculate ('mean', 'median', 'min', 'max', 'std').
    """
    # Get the unique labels from the labeled raster
    unique_labels = np.arange(0, labeled_raster.max() + 1)
    unique_labels = unique_labels[unique_labels != 0]

    # Initialize a dictionary to hold the results
    results = {label: {} for label in unique_labels}
    for stat in tqdm(stats, desc="Calculating statistics"):
        if stat == 'mean':
            values = ndimage.mean(base_raster, labels=labeled_raster, index=unique_labels)
        elif stat == 'count':
            values = region_count(labels=labeled_raster, index=unique_labels)
        elif stat == 'min':
            values = ndimage.labeled_comprehension(base_raster, labeled_raster, unique_labels, np.nanmin, float, np.nan)
        elif stat == 'max':
            values = ndimage.labeled_comprehension(base_raster, labeled_raster, unique_labels, np.nanmax, float, np.nan)
        elif stat == 'std':
            values = ndimage.standard_deviation(base_raster, labels=labeled_raster, index=unique_labels)
        elif stat == 'median':
            values = ndimage.labeled_comprehension(base_raster, labeled_raster, unique_labels, bn.nanmedian, float, np.nan)
            # values = ndimage.labeled_comprehension(
            #     base_raster, labeled_raster, unique_labels, lambda x: np.nanpercentile(x, 50), float, 0
            # )
        elif stat.endswith('p') and stat[:-1].isdigit():
            q = float(stat[:-1])
            values = ndimage.labeled_comprehension(
                base_raster, labeled_raster, unique_labels, lambda x: np.nanpercentile(x, q), float, 0
            )
        else:
            print(f"Statistic '{stat}' is not supported. Skipping.")

        for label, value in zip(unique_labels, values):
            results[label][stat] = value

    return results

def region_count(labels, index):
    counts = np.bincount(labels.ravel())
    # Handle case where some indices might be larger than max label
    result = np.zeros(len(index), dtype=int)
    valid_mask = index < len(counts)
    result[valid_mask] = counts[index[valid_mask]]
    return result
