"""
Parameter and mask management for configuration.
"""
import os
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union
from .. import parameter as pm


class ParameterManager:
    """
    Handles initialization and management of parameters and masks.
    """
    
    def __init__(self, config_instance):
        self.config = config_instance
        self.params = []
    
    def get_parameters_list(self):
        """
        Get the list of initialized parameters.
        
        Returns:
            List of Parameter objects
        """
        return self.params
    
    def add_parameter(self, array, thresholds=None, crs=None, transform=None, name=None, median_iterations=1, median_size=3):
        """
        Add a new parameter to the configuration.
        
        Args:
            array: The raster data as a numpy array
            crs: Coordinate Reference System
            transform: Affine transformation
            name: Parameter name
            thresholds: Threshold values
            median_iterations: Number of median filter iterations
            median_size: Size of median filter
        """
        if name is None:
            name = f"param{len(self.params) + 1}"
        
        param = pm.Parameter(
            self._check_name(name), 
            array=array, 
            crs=crs, 
            transform=transform, 
            thresholds=thresholds
        )
        param.median_config = {"size": median_size, "iterations": median_iterations}
        self.params.append(param)
    
    def add_mask(self, array=None, crs=None, transform=None, name=None, threshold=None):
        """
        Add a new mask to the configuration.

        Args:
            array: The raster data as a numpy array
            crs: Coordinate Reference System
            transform: Affine transformation
            name: Mask name
            threshold: Threshold value for the mask
        """
        if name is None:
            name = f"mask{len(self.params) + 1}"
        if isinstance(threshold, (list, tuple)):
            threshold = threshold[0]

        threshold = [threshold]
        mask = pm.Mask(
            self._check_name(name),
            array=array,
            crs=crs,
            transform=transform,
            threshold=threshold
        )
        self.params.append(mask)
    
    def config_array(self, param, crs, transform, mask=None):
        """
        Initialize configuration with array data and metadata.
        
        Args:
            param: Dictionary of parameters
            crs: Coordinate Reference System
            transform: Affine transformation
            mask: Optional mask dictionary
        """
        for key, value in param.items():
            if isinstance(value, tuple) and len(value) == 2:
                param_obj = pm.Parameter(
                    self._check_name(key), 
                    array=value[0], 
                    crs=crs, 
                    transform=transform, 
                    thresholds=value[1] if len(value) > 1 else None
                )
                self.params.append(param_obj)
            else:
                raise ValueError("Provide thresholds")
                
        if mask is not None:
            for key, value in mask.items():
                if isinstance(value, list):
                    mask_param = pm.Parameter(
                        self._check_name(key), 
                        array=value[0], 
                        crs=crs, 
                        transform=transform, 
                        thresholds=value[1] if len(value) > 1 else None
                    )
                    mask_param.mask = True
                    self.params.append(mask_param)
                else:
                    raise ValueError("Provide thresholds for mask")
    
    def init_parameters_from_config(self, param_file_dicts, mask_file_dicts):
        """
        Initialize parameters from configuration dictionaries.
        
        Args:
            param_file_dicts: Dictionary of parameter configurations
            mask_file_dicts: Dictionary of mask configurations
        """
        param_list = []
        
        # Initialize parameters
        for param_name, parameters in tqdm(param_file_dicts.items(), desc="Initializing parameters"):
            param_path = parameters.get('path', None)
            param_thresholds = parameters.get('thresholds', None)
            param_operator = parameters.get('operator', ">")
            median_config = parameters.get('median', {})

            param = pm.Parameter(name=param_name, raster_path=param_path, thresholds=param_thresholds)
            param.operator = param_operator
            param.median_config = median_config
            param_list.append(param)

        # Initialize masks
        for mask_name, parameters in tqdm(mask_file_dicts.items() if mask_file_dicts else {}, desc="Initializing masks"):
            mask_path = parameters.get('path', None)
            mask_thresholds = parameters.get('thresholds', None)
            mask_operator = parameters.get('operator', ">")
            mask_median = parameters.get('median', {})
            mask_keep_shape = parameters.get('keep_shape', False)

            mask_param = pm.Mask(name=mask_name, raster_path=mask_path, threshold=mask_thresholds)
            mask_param.operator = mask_operator
            mask_param.median = mask_median
            mask_param.keep_shape = mask_keep_shape
            param_list.append(mask_param)
        
        self.params = param_list
    
    def _check_name(self, name):
        """
        Check if the name is valid for the current driver.
        
        Args:
            name: Parameter or mask name
            
        Returns:
            Validated name
        """
        try:
            from .output_manager import OutputManager
            output_manager = OutputManager(self.config)
            
            if output_manager.get_driver() == "ESRI Shapefile":
                print("Using ESRI Shapefile driver, truncating name to 6 characters.")
                return name[:6]
            else:
                return name
        except (ValueError, KeyError):
            # If current process is not set or driver can't be determined, 
            # just return the name as-is
            return name
