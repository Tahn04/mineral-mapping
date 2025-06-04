import json
import os

import core.utils as utils
import numpy as np
import bottleneck as bn
from scipy.ndimage import generic_filter
from tqdm import tqdm
from collections import defaultdict

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
        self.indic_params = None

    def load_config(self):
        with open(self.config_path, "r") as f:
            return json.load(f)
        
    def get_thresholds(self, param):
        if self.indic_params is not None:
            if param in self.indic_params["Param"]:
                return self.indic_params["Param"][param]
            elif param in self.indic_params["Mask"]:
                return self.indic_params["Mask"][param]
        return self.config["PARAM_SETTINGS"][param]["thresholds"]
    
    def get_mask_threshold(self, param):
        return self.indic_params["Mask"][param]
    
    def get_num_median_filter(self, param):
        return self.config["PARAM_SETTINGS"][param]["num_median_filter"]
    
    def get_num_majority_filter(self, param):
        return self.config["PARAM_SETTINGS"][param]["num_majority_filter"]
    
    def get_num_boundary_clean(self, param):
        return self.config["PARAM_SETTINGS"][param]["num_boundary_clean"]
    
    def get_indicator_parameters(self, indicator): # Dict
        return self.config["INDIC_SETTINGS"][indicator]["Param"]
    
    def get_indicator_mask(self, indicator): # Dict
        return self.config["INDIC_SETTINGS"][indicator]["Mask"]

    def indicator_check(self, name):
        """
        Search for the indicator.
        """
        for indic, details in self.config["INDIC_SETTINGS"].items():
            if indic == name:
                self.indic_params = details
                return (details)
        return None
    
    def get_indicator_param_names(self):
        """
        Get a list of parameter names from the current indicator parameters.
        Returns the keys of the 'Param' dict if indic_params is set.
        """
        if self.indic_params is not None:
            return list(self.indic_params["Param"].keys())
        return None
    
    def get_indicator_mask_names(self):
        """
        Get a list of parameter names from the current indicator parameters.
        Returns the keys of the 'Param' dict if indic_params is set.
        """
        if self.indic_params is not None:
            return list(self.indic_params["Mask"].keys())
        return None



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

def median_kernel_filter(raster, size=3, iterations=1):
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

def list_vectorize(raster_list, profile, param):
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

def list_vectorize_dict(labeled_raster_list, stats_dict_list, param):
    """
    Vectorizes a list of labeled rasters using a list of stats dictionaries.
    """
    transform = param.get_transform()
    crs = param.get_crs()

    
    results = []
    for labeled_raster, stats_dict in tqdm(zip(labeled_raster_list, stats_dict_list), total=len(labeled_raster_list), desc="Vectorizing with stats"):
        result = utils.vectorize_dict(labeled_raster, stats_dict, transform, crs)
        results.append(result)

    return results
def list_zonal_stats(labeled_raster_list, param):

    thresholds = param.get_thresholds()
    transform = param.get_transform()

    x_res = transform[0]
    y_res = abs(transform[4]) # y res is negative for north-up images
    pixel_area = x_res * y_res

    index = 0
    results = []
    for labeled_raster in tqdm(labeled_raster_list, desc="Calculating zonal stats"):
        result = utils.zonal_stats(labeled_raster, param.raster, thresholds[index], pixel_area)
        results.append(result)
        index += 1
        
    return results

def list_label_clusters(raster_list):
    return [
        utils.label_clusters(raster)
        for raster in tqdm(raster_list, desc="Labeling clusters")
    ]