import json
import os

import core.utils as utils
import numpy as np
import bottleneck as bn
from scipy.ndimage import generic_filter
import numpy as np
import numba as nb
from tqdm import tqdm
from scipy.ndimage import iterate_structure
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import pandas as pd

#===========================================#
# Processing Functions
#===========================================#

def full_threshold(raster, thresholds):

    # thresholds = get_thresholds(param)
    # thresholds = Defaults().get_thresholds(param)

    results = []
    for t in tqdm(thresholds, desc="Applying thresholds"):
        result = utils.threshold(raster, t)
        results.append(result)
        
    return results

def median_kernel_filter(raster, iterations=1, size=3):
    def bn_nanmedian(arr):
        return bn.nanmedian(arr)
    for i in tqdm(range(iterations), desc="Applying median filter"):
        raster = generic_filter(raster, bn_nanmedian, size=size)
    return raster

def list_majority_filter(raster_list, iterations=1, size=3):
    return [
        utils.majority_filter(raster, size=size, iterations=iterations)
        #for raster in raster_list
        for raster in tqdm(raster_list, desc="Applying majority filter")
    ]

def list_boundary_clean(raster_list, iterations=1, radius=1):
    return [
        utils.boundary_clean(raster, iterations=iterations, radius=radius)
        #for raster in raster_list
        for raster in tqdm(raster_list, desc="Boundary cleaning")
    ]

def list_vectorize(raster_list, thresholds, crs, transform):
    """
    Vectorizes a list of rasters using a list of thresholds.
    """
    index = 0
    results = []
    for raster in tqdm(raster_list, desc="Vectorizing"):
        result = utils.vectorize(raster, thresholds[index], transform, crs)
        results.append(result)
        index += 1
        
    return results

def check_stats_dict(stats_dict):
    """
    Checks if the first-level keys are strings. If so, returns true.
    """
    return all(isinstance(key, str) for key in stats_dict.keys())


def restructure_stats(stats_dict_list):
    """
    Restructures a list of stats dictionaries into a single dictionary.
    """
    result = []
    num_thresholds = len(next(iter(stats_dict_list.values())))

    for _ in range(num_thresholds):
        result.append({})

    for param, stat_list in stats_dict_list.items():
        for i, stat_dict in enumerate(stat_list):
            if i < len(result):
                for idx, metrics in stat_dict.items():
                    for metric_name, value in metrics.items():
                        key_name = f"{param}_{metric_name}"

                        if idx not in result[i]:
                            result[i][idx] = {}
                        result[i][idx][key_name] = value
            else:
                print(f"Warning: Index {i} out of bounds for result list. Skipping.")

    return result

def list_vectorize_dict(labeled_raster_list, stats_dict_list, crs, transform):
    """
    Vectorizes a list of labeled rasters using a list of stats dictionaries.
    """
    results = []
    for labeled_raster, stats_dict in tqdm(zip(labeled_raster_list, stats_dict_list), total=len(labeled_raster_list), desc="Vectorizing with stats"):
        result = utils.vectorize_dict(labeled_raster, stats_dict, transform, crs)
        results.append(result)

    return results

def _zonal_stats_wrapper(args):
    labeled_raster, data_raster, threshold, pixel_area = args
    return utils.zonal_stats(labeled_raster, data_raster, threshold, pixel_area)

def list_zonal_stats(labeled_raster_list, param, max_workers=4):
    thresholds = param.get_thresholds()
    transform = param.get_transform()

    x_res = transform[0]
    y_res = abs(transform[4])  # y res is negative for north-up images
    pixel_area = x_res * y_res

    # Ensure the lists are the same length
    coverage_mask = param.coverage_mask()
    thresholds = [0] + thresholds  # Create a new list, don't mutate in place
    labeled_raster_list = [coverage_mask] + labeled_raster_list

    # Pair each labeled raster with its threshold
    tasks = [
        (labeled_raster, param.raster, threshold, pixel_area)
        for labeled_raster, threshold in zip(labeled_raster_list, thresholds)
    ]

    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_zonal_stats_wrapper, task): i
            for i, task in enumerate(tasks)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Zonal stats (parallel)"):
            idx = futures[future]
            results[idx] = future.result()

    return results
# def list_zonal_stats(labeled_raster_list, param):

#     thresholds = param.get_thresholds()
#     transform = param.get_transform()

#     x_res = transform[0]
#     y_res = abs(transform[4]) # y res is negative for north-up images
#     pixel_area = x_res * y_res

#     if len(labeled_raster_list) > len(thresholds):
#         coverage_mask = param.coverage_mask()
#         thresholds.insert(0, 0)  # Insert a default threshold if needed
#         labeled_raster_list.insert(0, coverage_mask)  # Insert the mask at the beginning of the list

#     index = 0
#     results = []
#     for labeled_raster in tqdm(labeled_raster_list, desc="Calculating zonal stats"):
#         result = utils.zonal_stats(labeled_raster, param.raster, thresholds[index], pixel_area)
#         results.append(result)
#         index += 1
        
#     return results

def list_vectorize_stats(raster_list, stats_dict_list, profile, param):
    """
    Vectorizes a list of rasters using a list of stats dictionaries.
    """
    transform = profile.transform
    crs = profile.crs

    index = 0
    results = []
    for raster in tqdm(raster_list, desc="Vectorizing with stats"):
        result = utils.vectorize_stats(raster, stats_dict_list[index], transform, crs)
        results.append(result)
        index += 1
        
    return results

def list_zonal_stats2(polygons, param_list, crs, transform):
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
    # for polygon, param in tqdm(zip(polygons, param_list), total=len(polygons), desc="Calculating zonal stats"):
    #     result = utils.zonal_stats2(polygon, param.raster, param.value, param.pixel_area, crs, transform)
    #     results.append(result)
    results = gpd.GeoDataFrame()
    for param in param_list:
        temp = utils.zonal_stats2(polygons, param.raster, pixel_area, crs, transform)
        results = pd.concat([results, temp], ignore_index=True)
    return results

def list_label_clusters(raster_list, min_cluster_size=9):
    labeled_rasters = [
        utils.label_clusters(raster)
        for raster in tqdm(raster_list, desc="Labeling clusters")
    ]
    return labeled_rasters