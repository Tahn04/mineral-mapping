import os
import re


import core.utils as utils
import core.processing as pr
import numpy as np

"""
Would like an over wrapping class that handles the full vectorization process for a specific tile.
This would include: 
- parameters 
- indicators (handle parameteTracking)
"""

class ParameterTracking:
    """
    A class to track which parameters have been processed.
    
    Attributes:
    -----------
        tile_id (int): The ID of the tile being processed.
        raw: bool: Whether the parameters are raw or processed.
        parameters (list): List of parameters after generalized.
    """
    def __init__(self):
        """Initialize the ParameterTracking class."""
        self.tile_id = None
        self.median_filter = []
        self.parameters = {}

    def add_parameter(self, param, data):
        """Add a parameter to the tracking list."""
        if param not in self.parameters:
            self.parameters[param] = data

    def get_parameters(self):
        """Return the list of tracked parameters."""
        return self.parameters
    
    def add_median_filter(self, median_filter): # this need to be able to handle multiple median filters
        """Set the median filter for the tracked parameters."""
        self.median_filter.append(median_filter)
    
class Raster:
    """
    A class to handle raster data for a specific tile.

    Attributes:
    -----------
        path (str): The file path to the raster data.
        dataset: The opened raster dataset.
        raster: The raster band data.
    
    """
    def __init__(self, path):
        self.path = path
        self.dataset = utils.open_raster(path)
        self.raster = utils.open_raster_band(self.dataset, 1)

    def get_data(self):
        return self.raster
    
    def get_profile(self):
        return self.dataset.profile 

    def close(self):
        self.dataset.close()

class TileParameterization:  
    """
    A class to handle the full parameterizaion process for a specific tile.
    Can only handle one parameter at a time.
    Attributes:
    -----------
        path (str): The file path to the raster data.
        param (str): The parameter to be processed. 
    
    """
    def __init__(self, input_path, output_path, param=None): 
        self.input_path = input_path
        self.output_path = output_path
        self.raster = Raster(input_path).get_data()
        self.profile = Raster(input_path).get_profile()
        self.param_tracking = ParameterTracking()
        self.defaults = pr.Defaults() 
        if param is not None:
            self.param = param
        else:
            self.param = None

    def threshold(self, param):
        # if len(self.param_tracking.median_filter) is 0:
        #     n = self.defaults.get_num_median_filter(param)
        #     median_filtered = utils.median_kernel_filter(self.raster, size=3, iterations=n)
        # else:
        #     median_filtered = self.param_tracking.median_filter

        n = self.defaults.get_num_median_filter(param)
        median_filtered = utils.median_kernel_filter(self.raster, size=3, iterations=n)
        
        thresholds = self.defaults.get_thresholds(param)
        return pr.full_threshold(median_filtered, thresholds)

    def process(self):
        print(f"Processing parameter: {self.param}")
        
        thresholds = self.threshold(self.param)
        utils.show_raster(thresholds[0], cmap='gray', title=f"Boundary Cleaned: {self.param}")
       
        num_majority_filter = self.defaults.get_num_majority_filter(self.param)
        maj_filt_list = pr.list_majority_filter(thresholds, num_majority_filter)
        
        # Clean boundaries
        num_boundary_clean = self.defaults.get_num_boundary_clean(self.param)
        boundary_clean_list = pr.list_boundary_clean(maj_filt_list, num_boundary_clean)
        
        utils.show_raster(boundary_clean_list[0], cmap='gray', title=f"Boundary Cleaned: {self.param}")

        # Vectorize the cleaned boundaries
        vector_list = pr.list_vectorize(boundary_clean_list, self.profile, self.param)
        vector_stack = utils.merge_polygons(vector_list)

        utils.save_shapefile(vector_stack, self.output_path, "first.shp")
    
def get_file(path, param):
    """Get a file from the directory that matches the parameter."""
    files = os.listdir(path)
    pattern = re.compile(r"T1250_cdodtot_BAL1_(.+)\.IMG")
    for f in files:
        match = pattern.match(f)
        if match and match.group(1) == param:
            return os.path.join(path, f)
    return None


# full_raster = utils.open_raster(path)
# raster = utils.open_raster_band(full_raster, 1)
# # utils.show_raster(raster.read(1, masked=True), cmap='gray', title=None)

# """
# Median filter
# """
# meduian_filter = utils.median_kernel_filter(raster, size=3)
# # utils.show_raster(meduian_filter, cmap='gray', title=None)

# # print(np.nanmin(meduian_filter), np.nanmax(meduian_filter), np.nanmean(meduian_filter), np.nanstd(meduian_filter))

# """
# Thresholding
# """
# thresh_list = preprocessing.full_threshold(raster, "D2300")
# # utils.show_raster(thresh_raster_list[4], cmap='gray', title=None) # 0.015

# """
# Majority filter
# """
# maj_filt_list = preprocessing.list_majority_filter(thresh_list)
# # utils.show_raster(maj_filt_list[4], cmap='gray', title=None) # 0.015

# """
# Boundary clean
# """
# boundary_clean_list = preprocessing.list_boundary_clean(maj_filt_list)
# # utils.show_raster(boundary_clean_list[4], cmap='gray', title=None) # 0.015

# """
# Vectorize
# """
# vector_list = preprocessing.list_vectorize(boundary_clean_list, full_raster, "D2300")
# vector_stack = utils.merge_polygons(vector_list)
# # utils.show_polygons(vector_stack, title=None) # 0.015

# utils.save_shapefile(vector_stack, r'Code\mineral-mapping\outputs', "vectorized_output.shp")