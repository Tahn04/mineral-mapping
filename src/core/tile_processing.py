import os
import re


import core.utils as utils
import core.processing as pr
import numpy as np

class Parameter:
    def __init__(self, name: str, raster_path: str):
        self.name = name
        self.raster_path = raster_path
        self.dataset = utils.open_raster(raster_path)
        self.raster = utils.open_raster_band(self.dataset, 1)
        self.defaults = pr.Defaults()

    def coverage_mask(self):
        """Calculate the coverage mask for the parameter (True where raster is not NaN)."""
        return ~np.isnan(self.raster)

    def get_transform(self):
        """Return the affine transform of the raster dataset."""
        return self.dataset.transform

    def get_crs(self):
        """Return the coordinate reference system of the raster dataset."""
        return self.dataset.crs

    def get_thresholds(self):
        """Return the thresholds for the parameter."""
        return self.defaults.get_thresholds(self.name)

    def get_num_median_filter(self):
        """Return the number of median filter iterations for the parameter."""
        return self.defaults.get_num_median_filter(self.name)

    def get_num_majority_filter(self):
        """Return the number of majority filter iterations for the parameter."""
        return self.defaults.get_num_majority_filter(self.name)

    def get_num_boundary_clean(self):   
        """Return the number of boundary clean iterations for the parameter."""
        return self.defaults.get_num_boundary_clean(self.name)

    def close(self):
        """Close the raster dataset."""
        if self.dataset:
            self.dataset.close()
            self.dataset = None
            self.raster = None

class TileParameterization:  
    """
    A class to handle the full parameterizaion process for a specific tile.
    Can only handle one parameter at a time.
    Attributes:
    -----------
        path (str): The file path to the raster data.
        param (str): The parameter to be processed. 
    
    """
    def __init__(self, input_path, output_path, param): 
        self.input_path = input_path
        self.output_path = output_path
        self.parameter = Parameter(param, input_path)
        # self.profile = Raster(input_path)
        self.defaults = pr.Defaults() 
        if param is not None:
            self.param = param
        else:
            self.param = None

    def threshold(self):
        # if len(self.param_tracking.median_filter) is 0:
        #     n = self.defaults.get_num_median_filter(param)
        #     median_filtered = utils.median_kernel_filter(self.raster, size=3, iterations=n)
        # else:
        #     median_filtered = self.param_tracking.median_filter

        num_median_filter = self.defaults.get_num_median_filter(self.parameter.name)
        median_filtered = pr.median_kernel_filter(self.parameter.raster, size=3, iterations=num_median_filter)

        thresholds = self.defaults.get_thresholds(self.parameter.name)
        return pr.full_threshold(median_filtered, thresholds)

    def process(self):
        print(f"Processing parameter: {self.parameter.name}")

        thresholded = self.threshold()

        num_majority_filter = self.parameter.get_num_majority_filter()
        num_boundary_clean = self.parameter.get_num_boundary_clean()

        # first filters
        maj_filt_list = pr.list_majority_filter(thresholded, num_majority_filter)
        boundary_clean_list = pr.list_boundary_clean(maj_filt_list, num_boundary_clean)

        # second filters
        maj3_filt_list = pr.list_majority_filter(boundary_clean_list, 3)
        sieve_list = utils.list_sieve_filter(maj3_filt_list, profile=self.parameter)
        boundary_clean_list = pr.list_boundary_clean(sieve_list, num_boundary_clean)

        labeled_raster_list = pr.list_label_clusters(boundary_clean_list)
        zonal_stats = pr.list_zonal_stats(labeled_raster_list, self.parameter)
        vector_list = pr.list_vectorize_dict(labeled_raster_list, zonal_stats, self.parameter)
        vector_stack = utils.merge_polygons(vector_list)

        utils.save_shapefile(vector_stack, self.output_path, "D2300_full_ZS3.shp")

# def get_file(path, param):
#     """Get a file from the directory that matches the parameter."""
#     files = os.listdir(path)
#     pattern = re.compile(r"T1250_cdodtot_BAL1_(.+)\.IMG")
#     for f in files:
#         match = pattern.match(f)
#         if match and match.group(1) == param:
#             return os.path.join(path, f)
#     return None

# class TileParameterizationFactory:
#     """
#     A factory class to create TileParameterization instances based on the parameter.
    
#     Attributes:
#     -----------
#         input_path (str): The file path to the raster data.
#         output_path (str): The directory where output files will be saved.
#     """
#     def __init__(self, input_path, output_path):
#         self.input_path = input_path
#         self.output_path = output_path

#     def create(self, param):
#         """Create a TileParameterization instance for the given parameter."""
#         file_path = get_file(self.input_path, param)
#         if file_path:
#             return TileParameterization(file_path, self.output_path, param)
#         else:
#             raise ValueError(f"No file found for parameter: {param}")
        


# class Raster:
#     """
#     A class to handle raster data for a specific tile.

#     Attributes:
#     -----------
#         path (str): The file path to the raster data.
#         dataset: The opened raster dataset.
#         raster: The raster band data.
    
#     """
#     def __init__(self, path):
#         self.path = path
#         self.dataset = utils.open_raster(path)
#         self.raster = utils.open_raster_band(self.dataset, 1)
#         self.transform = self.dataset.transform
#         self.crs = self.dataset.crs

#     def get_data(self):
#         """Return the raster data."""
#         return self.raster
    
#     def get_profile(self):
#         return self.dataset.profile 

#     def close(self):
#         self.dataset.close()
# class ParameterTracking:
#     """
#     A class to track which parameters have been processed.
    
#     Attributes:
#     -----------
#         tile_id (int): The ID of the tile being processed.
#         raw: bool: Whether the parameters are raw or processed.
#         parameters (list): List of parameters after generalized.
#     """
#     def __init__(self):
#         """Initialize the ParameterTracking class."""
#         self.tile_id = None
#         self.median_filter = []
#         self.parameters = {}

#     def add_parameter(self, param, data):
#         """Add a parameter to the tracking list."""
#         if param not in self.parameters:
#             self.parameters[param] = data

#     def get_parameters(self):
#         """Return the list of tracked parameters."""
#         return self.parameters
    
#     def add_median_filter(self, median_filter): # this need to be able to handle multiple median filters
#         """Set the median filter for the tracked parameters."""
#         self.median_filter.append(median_filter)