import json
import os

from . import utils
import numpy as np
from tqdm import tqdm

class Raster:
    def __init__(self, path):
        self.path = path
        self.dataset = utils.open_raster(path)
        self.raster = utils.open_raster_band(self.dataset, 1)

    def get_data(self):
        return self.raster.read(1, masked=True)

    def close(self):
        self.dataset.close()

def get_thresholds(param):
    """
    Get thresholds from a JSON config file.

    Args:
        param (str): The parameter to get thresholds for.

    Returns:
        list: List of thresholds.
    """
    # Load thresholds from JSON config
    config_path = os.path.join("Code", "mineral-mapping", "config", "default.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    thresholds = config["PARAM_SETTINGS"][param]["thresholds"]  # Adjust key as needed
    return thresholds

def full_threshold(raster, param):
    """
    Apply a threshold to a raster dataset.

    Args:
        raster (numpy.ndarray): The raster data.
        threshold (float): The threshold value.

    Returns:
        numpy.ndarray: The thresholded raster data.
    """
    thresholds = get_thresholds(param)

    results = []
    for t in tqdm(thresholds, desc="Applying thresholds"):
        result = utils.threshold(raster, t)
        results.append(result)
        
    return results

def list_majority_filter(raster_list):
    """
    Apply a majority filter to a list of rasters.

    Args:
        raster_list (list): List of raster data.
        size (int): Size of the kernel.

    Returns:
        list: List of filtered raster data.
    """
    results = []
    for raster in tqdm(raster_list, desc="Majority filtering"):
        result = utils.majority_filter(raster)
        results.append(result)
        
    return results

def list_boundary_clean(raster_list):
    """
    Apply a boundary clean to a list of rasters.

    Args:
        raster_list (list): List of raster data.
        size (int): Size of the kernel.

    Returns:
        list: List of filtered raster data.
    """
    results = []
    for raster in tqdm(raster_list, desc="Boundary cleaning"):
        result = utils.boundary_clean(raster)
        results.append(result)
        
    return results

def list_vectorize(raster_list, profile, param):
    """
    Apply a vectorization to a list of rasters.

    Args:
        raster_list (list): List of raster data.
        size (int): Size of the kernel.

    Returns:
        list: List of filtered raster data.
    """
    transform = profile.transform
    crs = profile.crs

    thresholds = get_thresholds(param)

    index = 0
    results = []
    for raster in tqdm(raster_list, desc="Vectorizing"):
        result = utils.vectorize(raster, thresholds[index], transform, crs)
        results.append(result)
        index += 1
        
    return results
