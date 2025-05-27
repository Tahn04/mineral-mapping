import json
import os

import core.utils as utils
import numpy as np
from tqdm import tqdm

def get_thresholds(param):
    config_path = os.path.join("Code", "mineral-mapping", "config", "default.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    thresholds = config["PARAM_SETTINGS"][param]["thresholds"]  # Adjust key as needed
    return thresholds

def get_num_median_filter(param):
    config_path = os.path.join("Code", "mineral-mapping", "config", "default.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    num_median_filter = config["PARAM_SETTINGS"][param]["num_median_filter"]  # Adjust key as needed
    return num_median_filter

class Defaults:
    """
    Class to hold default parameters for processing.
    """
    def __init__(self):
        self.config_path = os.path.join("Code", "mineral-mapping", "config", "default.json")
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            return json.load(f)
        
    def get_thresholds(self, param):
        return self.config["PARAM_SETTINGS"][param]["thresholds"]
    
    def get_num_median_filter(self, param):
        return self.config["PARAM_SETTINGS"][param]["num_median_filter"]
    
    def get_num_majority_filter(self, param):
        return self.config["PARAM_SETTINGS"][param]["num_majority_filter"]
    
    def get_num_boundary_clean(self, param):
        return self.config["PARAM_SETTINGS"][param]["num_boundary_clean"]
    
    def get_indicator_parameters(self, indicator): # Dict
        return self.config["PARAM_SETTINGS"][indicator]["Param"]
    
    def get_indicator_mask(self, indicator): # Dict
        return self.config["PARAM_SETTINGS"][indicator]["Mask"]

def full_threshold(raster, thresholds):

    # thresholds = get_thresholds(param)
    # thresholds = Defaults().get_thresholds(param)

    results = []
    for t in tqdm(thresholds, desc="Applying thresholds"):
        result = utils.threshold(raster, t)
        results.append(result)
        
    return results

def list_majority_filter(raster_list, iterations=1):

    results = []
    for raster in tqdm(raster_list, desc="Majority filtering"):
        result = utils.majority_filter(raster)
        for _ in range(iterations - 1):
            result = utils.majority_filter(result)
        results.append(result)
        
    return results

def list_boundary_clean(raster_list, iterations=1):
    results = []
    for raster in tqdm(raster_list, desc="Boundary cleaning"):
        result = utils.boundary_clean(raster)
        for _ in range(iterations - 1):
            result = utils.boundary_clean(result)
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
