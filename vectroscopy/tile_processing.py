import os
import re
import json

# from . import config as cfg
from . import raster_ops as ro
from . import vector_ops as vo
from . import parameter as pm
from . import file_handler as fh
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shapely
import xarray as xr

class ProcessingPipeline:
    """
    A class to handle the complete processing pipeline dictated by a YAML file.
    
    Attributes:
    -----------
        yaml_file (str): The path to the YAML file containing the processing configuration.
    """
    def __init__(self, config):
        self.config = config
        self.crs = None
        self.transform = None
        self.mask = None
        self.indicator = False
    
    def process_file(self):
        """
        Process the parameter or indicator based on the name.
        """
        # for process_name, process in tqdm(self.config.processes.items(), desc="Processing Processes"):
        try:
            fh.FileHandler()
            print(fh.FileHandler().get_directory())
            process = self.config.get_current_process()
            for _ in tqdm(range(1), desc=f"Processing: {process['name']}"):
                param_list = self.config.get_parameters_list()
                processed_rasters = self.process_parameters(param_list)
                return self.vectorize(processed_rasters, param_list)
        finally:
            fh.FileHandler().cleanup()
            print("files cleaned up.")

    def vectorize(self, raster_list, param_list):
        """
        Vectorize the raster data based on the zonal statistics.
        
        Args:
            process: The process configuration dictionary.
            raster_list: A list of processed raster data.
            zonal_stats: The zonal statistics for the raster data.
        
        Returns:
            List: A list of vectorized geometries.
        """
        simplification_level = self.config.get_simplification_level() 
        driver = self.config.get_driver()
        thresholds = self.assign_thresholds(raster_list, param_list)
        stats_list = self.config.get_stats()

        start_old = time.time()
        # file_paths = vo.list_raster_to_shape_gdal(raster_list, thresholds, self.crs, self.transform)
        # gdf = vo.list_file_zonal_stats(file_paths, param_list, self.crs, self.transform, stats_list, simplification_level)
        gdf = vo.list_vectorize(raster_list, thresholds, self.crs, self.transform, simplification_level)
        gdf = vo.list_zonal_stats(gdf, param_list, self.transform, stats_list)
        end_old = time.time()
        print(f"Old vectorization took {end_old - start_old:.2f} seconds")
        color = self.config.get_color()
        if color:    
            gdf = self.assign_color(gdf, color=color)

        gdf.set_crs(self.crs, inplace=True)
        cs = self.config.get_cs(self.crs)
        projected_gdf = gdf.to_crs(cs) 

        projected_gdf.geometry = shapely.set_precision(projected_gdf.geometry, grid_size=0.00001) # slow 
        # projected_gdf.geometry = projected_gdf.geometry.se
        if driver == "pandas":
            return projected_gdf
        
        self.save_gdf(projected_gdf, driver)
        return None

    def save_gdf(self, gdf, driver):
        """
        Save the GeoDataFrame to a file.
        """
        stack = self.config.get_stack()
        output_dict = self.config.get_output_path()
        name = self.config.get_current_process()["name"]
        if stack is False:
            thresholds = gdf['Threshold'].unique()
            if len(thresholds) > 1:
                new_output_dict = os.path.join(output_dict, str(name))
                os.makedirs(new_output_dict, exist_ok=True)
                for threshold in thresholds:
                    thresh_gdf = gdf[gdf['Threshold'] == threshold]
                    filename = fh.FileHandler().create_output_filename(driver, name, threshold)
                    vo.save_gdf_to_file(thresh_gdf, new_output_dict, filename, driver=driver)
                return None
        filename = fh.FileHandler().create_output_filename(driver, name, "stack")
        vo.save_gdf_to_file(gdf, output_dict, filename, driver=driver)

    def process_parameters(self, param_list):
        """
        Process the raster data based on the configuration.

        Returns:
            List: A list of processed raster data
        """
        start_time = time.time()

        param_list, post_processing_masks = self.get_post_processing_masks(param_list)
        full_mask = self.complete_mask(param_list, post_processing_masks)
        raster_list = self.threshold(param_list, full_mask)
        
        end_time = time.time()
        print(f"xarray_to_array execution time: {end_time - start_time:.2f} seconds")
        
        target_param = param_list[0]
        self.crs = target_param.crs
        self.transform = target_param.transform

        raster_list = self.processing_pipeline(raster_list)

        start_time = time.time()
        raster_list = self.clean_rasters(raster_list, param_list, full_mask)
        end_time = time.time()
        print(f"clean_rasters execution time: {end_time - start_time:.2f} seconds")
        return raster_list

    def clean_rasters(self, raster_list, param_list, full_mask):
        """Applies the coverage mask to the raster list - optimized version."""
        # Early exit for empty inputs
        if not raster_list:
            return []

        coverage_mask = full_mask

        # Convert coverage_mask to numpy once if it's xarray
        if isinstance(coverage_mask, xr.DataArray):
            coverage_mask_np = coverage_mask.values  # Use .values instead of .data.compute()
        else:
            coverage_mask_np = coverage_mask
        
        # Pre-allocate result list
        base_check = self.config.get_base_check()
        final_raster_list = []
        if base_check:
            final_raster_list.append(coverage_mask_np.astype(np.uint8))
        
        # Process rasters in-place when possible
        for raster in tqdm(raster_list, desc="Cleaning rasters"):
            if isinstance(raster, np.ndarray):
                # Direct numpy operation - fastest path
                cleaned = raster * coverage_mask_np
            else:
                # Handle xarray case
                if hasattr(raster, 'values'):
                    cleaned = raster.values * coverage_mask_np
                else:
                    # Fallback for other array types
                    cleaned = np.asarray(raster) * coverage_mask_np
            
            final_raster_list.append(cleaned)
        
        return final_raster_list

    @staticmethod
    def clip_raster(raster, mask, val=1):
        """Masks the raster data using a boolean mask or a value - optimized."""
        # Create boolean mask efficiently
        if val:
            if isinstance(mask, xr.DataArray):
                bool_mask = mask != val
            else:
                bool_mask = mask != val
        else:
            bool_mask = mask
        
        # Apply mask directly without logical_not operations
        if isinstance(raster, xr.DataArray) and isinstance(bool_mask, xr.DataArray):
            return raster.where(~bool_mask, 0)  # Use xarray's where method
        else:
            # Convert to numpy for efficient operations
            if isinstance(raster, xr.DataArray):
                raster_data = raster.values
            else:
                raster_data = raster
                
            if isinstance(bool_mask, xr.DataArray):
                mask_data = bool_mask.values
            else:
                mask_data = bool_mask
            
            return raster_data * ~mask_data

    def calculate_param_coverage_mask(self, param_list):
        """Calculate the coverage mask for the raster data - optimized."""
        if not param_list:
            return None
        
        # Handle single parameter case efficiently
        if len(param_list) == 1:
            param = param_list[0]
            if isinstance(param, xr.DataArray):
                return ~np.isnan(param.values)  # Use .values instead of .data
            else:
                return param.coverage_mask()
        
        # For multiple parameters, use more efficient reduction
        masks = []
        for param in param_list:
            if isinstance(param, xr.DataArray):
                mask = ~np.isnan(param.values)
            else:
                mask = param.coverage_mask()
            masks.append(mask)
        
        # Use numpy's logical_and.reduce for efficient multi-mask combination
        return np.logical_and.reduce(masks)
    
    def complete_mask(self, param_list, post_processing_masks=None):
        """Calculate the coverage mask for the raster data."""
        coverage_mask = self.calculate_param_coverage_mask(param_list)
   
        if post_processing_masks:
            for mask in post_processing_masks:
                val = mask.thresholds[0]
                coverage_mask = self.clip_raster(coverage_mask, mask.dataset, val=val)
        return coverage_mask

    # Alternative version using lazy evaluation for very large datasets
    def clean_rasters_lazy(self, raster_list, param_list, post_processing_masks):
        """Memory-efficient version using generators for large datasets."""
        if not raster_list:
            return []
        
        coverage_mask = self.calculate_param_coverage_mask(param_list)
        
        # Apply post-processing masks
        if post_processing_masks:
            for mask in post_processing_masks:
                val = mask.thresholds[0]
                coverage_mask = self.clip_raster(coverage_mask, mask.raster, val=val)
        
        # Convert to numpy once
        if isinstance(coverage_mask, xr.DataArray):
            coverage_mask_np = coverage_mask.values
        else:
            coverage_mask_np = coverage_mask
        
        def process_raster(raster):
            """Process a single raster with the coverage mask."""
            if isinstance(raster, np.ndarray):
                return raster * coverage_mask_np
            elif hasattr(raster, 'values'):
                return raster.values * coverage_mask_np
            else:
                return np.asarray(raster) * coverage_mask_np
        
        # Build result list
        final_raster_list = []
        base_check = self.config.get_base_check()
        if base_check:
            final_raster_list.append(coverage_mask_np.astype(np.uint8))
        
        # Process rasters one by one to save memory
        for raster in tqdm(raster_list, desc="Cleaning rasters"):
            final_raster_list.append(process_raster(raster))
        
        return final_raster_list
    # def clean_rasters(self, raster_list, param_list, post_processing_masks):
    #     """Applies the coverage mask to the raster list."""
    #     coverage_mask = self.calculate_coverage_mask(param_list)

    #     for mask in post_processing_masks:
    #         val=mask.get_thresholds()[0]
    #         coverage_mask = self.clip_raster(coverage_mask, mask.raster, val=val)

    #     for i in range(len(raster_list)):
    #         raster_list[i] = xr.DataArray(raster_list[i], dims=('y', 'x')) * coverage_mask
    #     raster_list = list(raster_list)

    #     base_check = self.config.get_base_check()
    #     if base_check:
    #         raster_list.insert(0, coverage_mask.astype(np.uint8))
    #     final_raster_list = []
    #     for raster in tqdm(raster_list, desc="Cleaning rasters"):
    #         if isinstance(raster, xr.DataArray):
    #             if hasattr(raster.data, 'compute'):
    #                 array = raster.data.compute()
    #             else:
    #                 array = raster.data
    #             final_raster_list.append(np.asarray(array))
    #     return final_raster_list

    # @staticmethod
    # def clip_raster(raster, mask, val=1):
    #     """Masks the raster data using a boolean mask or a value."""
    #     if val:
    #         mask = (mask != val)
    #     if isinstance(mask, xr.DataArray) and isinstance(raster, xr.DataArray):
    #         mask =  xr.ufuncs.logical_not(mask)
    #     else:
    #         mask = np.logical_not(mask)
    #     return raster * mask
    
    # def calculate_coverage_mask(self, param_list):
    #     """Calculate the coverage mask for the raster data."""
    #     # Multiply all coverage masks together to get a joint coverage mask
    #     joint_mask = None
    #     for param in param_list:
    #         if isinstance(param, xr.DataArray):
    #             coverage_mask = ~np.isnan(param.data)
    #         else:
    #             coverage_mask = param.coverage_mask()
    #         if joint_mask is None:
    #             joint_mask = coverage_mask
    #         else:
    #             joint_mask = joint_mask & coverage_mask
    #     return joint_mask
            
    
    def get_post_processing_masks(self, param_list):
        """ Get a list of parameters that will be cut after processing."""
        keep_shape_masks = []
        # Iterate over a copy of param_list to safely remove items while iterating
        for param in param_list[:]:
            if isinstance(param, pm.Mask) and param.keep_shape:
                keep_shape_masks.append(param)
                param_list.remove(param)
        return param_list, keep_shape_masks

    def threshold(self, param_list, mask=None):
        """
        Applies median filter, thresholds, and then masks the data.
        
        Args:
            process: The process configuration dictionary.
            param_list: A list of Parameter objects initialized with the raster data.
        
        Returns:
            List: A list of processed raster data at the number of desired intervals.
        """
        param_thresholded_list = []
        masks_thresholded_list = []
        for param in param_list:
            median_iterations = param.get_median_config().get("iterations", 0)
            median_size = param.get_median_config().get("size", 3)
            # Apply median filter
            start = time.time()
            preprocessed = ro.dask_nanmedian_filter(param.dataset, median_size, median_iterations)
            mid = time.time()
            print(f"Median filter execution time: {mid - start:.2f} seconds")

            if mask is not None:
                preprocessed = preprocessed.where(mask, np.nan)
            self.save_raster(preprocessed, param)
            final = time.time()
            print(f"Raster saved execution time: {final - mid:.2f} seconds")
        
            if param.mask:
                masks_thresholded_list.append(param.threshold(preprocessed, param.thresholds))
            else:
                param_thresholded_list.append(param.threshold(preprocessed, param.thresholds))

        return self.xarray_combine_thresholded_rasters(
            param_thresholded_list,
            masks_thresholded_list
        )
        # # Combine the thresholded rasters
        # if len(masks_thresholded_list) > 0 or len(param_thresholded_list) > 1:
        #     if isinstance(param_thresholded_list[0], xr.DataArray):
        #         return ro.combine_thresholded_rasters_detailed(
        #             param_thresholded_list, 
        #             masks_thresholded_list,
        #         )
        #     else:
        #         self.indication = True
        #         param_levels = list(zip(*param_thresholded_list))

        #         combined_mask = np.logical_not(np.logical_or.reduce(masks_thresholded_list)).astype(np.uint32)
                
        #         if combined_mask.ndim == 3 and combined_mask.shape[0] == 1: # Check if alwasy true
        #             combined_mask = np.squeeze(combined_mask, axis=0)
        #         # ro.show_raster(combined_mask, title="mask")
        #         raster_list = [
        #             np.prod(level_rasters, axis=0)
        #             for level_rasters in param_levels
        #         ]
        #         # ro.show_raster(raster_list[0], title="threshold - Processed Raster lowest")
        #         for i in range(len(raster_list)):
        #             if raster_list[i].ndim == 3 and raster_list[i].shape[0] == 1: # Check if alwasy false
        #                 raster_list[i] = np.squeeze(raster_list[i], axis=0)
        #             raster_list[i] = raster_list[i] * combined_mask
        #         # ro.show_raster(raster_list[0], title="threshold - Processed Raster lowest")
        #         return raster_list
        
        # return param_thresholded_list[0]
    
    def xarray_combine_thresholded_rasters(self, param_thresholded, masks_thresholded=[]):
        """
        More detailed version with explicit handling of dimensions and metadata.
        """
        
        if len(masks_thresholded) > 0 or len(param_thresholded) > 1:
            self.indicator = True
            if param_thresholded:
                threshold_coords = param_thresholded[0].coords['threshold']
                template = param_thresholded[0]
            else:
                threshold_coords = masks_thresholded[0].coords['threshold']
                template = masks_thresholded[0]
            
            # Process masks (subtractive)
            if len(masks_thresholded) > 0:
                # Combine all masks using logical OR
                combined_mask_raw = masks_thresholded[0].copy() 
                for mask in masks_thresholded[1:]: # should make sure they are 2D
                    combined_mask_raw =  xr.ufuncs.logical_or(combined_mask_raw, mask[0])
                
                # Invert mask (areas to keep)
                combined_mask = xr.ufuncs.logical_not(combined_mask_raw)
            else:
                # No masks - create all-ones mask
                combined_mask = xr.ones_like(template, dtype='uint32') # ???
            
            # Process parameters (multiplicative)
            if len(param_thresholded) > 1:
                param_thresholded = self.assign_thresholds_to_params(param_thresholded)
                combined_params = param_thresholded[0].copy()
                for param in param_thresholded[1:]:
                    combined_params = combined_params * param
            else:
                combined_params = param_thresholded[0]
            
            # Apply mask
            result = combined_params * combined_mask[0]
            
            # Update metadata
            result.attrs['operation'] = 'combined_thresholded_rasters'
            result.attrs['num_masks'] = len(masks_thresholded)
            result.attrs['num_params'] = len(param_thresholded)
            
            return result
        else:
            return param_thresholded[0]
        
    def save_raster(self, data, param):
        """
        Save the raster data to a file only if stats are to be computed.
        """
        if self.config.get_stats():
            preprocessed_path = fh.FileHandler().create_temp_file(
                "preprocessed", 
                "tif",
            )
            ro.save_raster_fast_rasterio(
                data,
                param.crs,
                param.transform,
                preprocessed_path
            )
            param.preprocessed_path = preprocessed_path

    @staticmethod
    def assign_thresholds_to_params(param_thresholded):
        num_thresholds = len(param_thresholded[0].threshold)
        if num_thresholds > 0:
            threshold_list = list(range(1, num_thresholds + 1))
            new_param_thresholded = []
            for param in param_thresholded:
                param = param.assign_coords(threshold=threshold_list)
                new_param_thresholded.append(param)
        return new_param_thresholded
    
    def processing_pipeline(self, raster_list):
        """
        Execute the processing pipeline.
        """
        show_rasters = False
        if show_rasters:
            ro.show_raster(raster_list[0], title="threshold- Processed Raster lowest")
            # utils.save_raster(raster_list[0], r"\\lasp-store\home\taja6898\Documents\Mars_Data\T1250_demo_parameters", "MC13_thresholded_0.tif", param_list[0].dataset.profile)
        for task in self.config.get_pipeline() if self.config.get_pipeline() else []:
            task_name = task.get("task", "")
            if "majority" in task_name:
                iterations = self.config.get_task_param(task, "iterations")
                size = self.config.get_task_param(task, "size")    

                iterations = 1 if iterations is None else iterations
                size = 3 if size is None else size
                raster_list = ro.dask_list_majority_filter(raster_list, iterations=iterations, size=size)
                if show_rasters:
                    ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")

            elif "boundary" in task_name:
                iterations = self.config.get_task_param(task, "iterations")
                size = self.config.get_task_param(task, "size")

                iterations = 1 if iterations is None else iterations
                size = 3 if size is None else size
                raster_list = ro.list_boundary_clean(raster_list, iterations=iterations, radius=size)
                if show_rasters:
                    ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")

            elif "sieve" in task_name:
                threshold = self.config.get_task_param(task, "threshold")
                iterations = self.config.get_task_param(task, "iterations")
                connectedness = self.config.get_task_param(task, "connectedness")

                threshold = 9 if threshold is None else threshold
                iterations = 1 if iterations is None else iterations
                connectedness = 4 if connectedness is None else connectedness

                raster_list = ro.list_sieve_filter_rio(
                    raster_list,
                    iterations=iterations,
                    threshold=threshold,
                    connectedness=connectedness
                )
                if show_rasters:
                    ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")
            elif "open" in task_name:
                iterations = self.config.get_task_param(task, "iterations")
                size = self.config.get_task_param(task, "size")

                iterations = 1 if iterations is None else iterations
                size = 3 if size is None else size
                raster_list = ro.list_binary_opening(raster_list, iterations=iterations, size=size)
                if show_rasters:
                    ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")
        
        return raster_list

    def assign_thresholds(self, raster_list, param_list):
        """
        Assign thresholds to the raster data based on the parameters.
        
        Args:
            raster_list: A list of processed raster data.
            param_list: A list of Parameter objects initialized with the raster data.
        
        Returns:
            List: A list of thresholds for each parameter.
        """
        if self.indicator:
            size = len(raster_list)
            thresholds = [i + 1 for i in range(size)]
        else:
            thresholds = param_list[0].thresholds
        
        base_check = self.config.get_base_check()
        if base_check:
            thresholds.insert(0, 0)
        
        return thresholds

    def assign_color(self, gdf, color="viridis"):
        """
        Assign colors to the geometries in the GeoDataFrame based on the thresholds.

        Args:
            gdf: The GeoDataFrame containing the geometries.
            colormap (str): The name of the matplotlib colormap to use for coloring the geometries.

        Returns:
            GeoDataFrame: The input GeoDataFrame with an added 'color' column.
        """
        c = {'red': [255,0,0], 'orange': [255,128,0], 'yellow': [255,255,0],
            'lime': [128,255,0], 'green': [0,255,0], 'sea': [0,255,128],
            'cyan': [0,255,255], 'sky': [0,128,255], 'blue': [0,0,255],
            'violet': [128,0,255], 'magenta': [255,0,255], 'pink': [255,0,128]}
        
        thresholds = gdf['Threshold'].unique()
        
        if color in c:
            end_color = c[color]
            # Use a linear color ramp from white to the selected color, mapped by threshold value
            thresholds_sorted = sorted(thresholds)
            color_map = self.make_ramp(end_color, len(thresholds_sorted))
            threshold_to_color = {val: color_map[i][0] for i, val in enumerate(thresholds_sorted)}
            gdf['hex_color'] = gdf['Threshold'].map(threshold_to_color)

        elif color in plt.colormaps():
            cmap = plt.get_cmap(color, len(thresholds))
            color_map = {val: mcolors.to_hex(cmap(i)) for i, val in enumerate(sorted(thresholds))}
            gdf['hex_color'] = gdf['Threshold'].map(color_map)
        
        else:
            raise ValueError(f"Color '{color}' is not recognized. Use a valid matplotlib colormap name or a predefined color.")

        return gdf

    @staticmethod
    def make_ramp(end_color, num_thresh):
        r, g, b = end_color
        rs, gs, bs = (
            np.linspace(0, r, num_thresh+1),
            np.linspace(0, g, num_thresh+1),
            np.linspace(0, b, num_thresh+1))
        return [[ProcessingPipeline.rgb_to_hex(int(rs[i]), int(gs[i]), int(bs[i]))] for i in range(num_thresh+1)]

    @staticmethod
    def rgb_to_hex(r, g, b):
        """
        Converts RGB color values (0-255) to a hexadecimal color code.

        Args:
            r (int): Red component (0-255).
            g (int): Green component (0-255).
            b (int): Blue component (0-255).

        Returns:
            str: The hexadecimal color code in the format '#RRGGBB'.
        """
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
    def get_bool_masks(self, param_list):
        """ Get a list of parameters that will be cut after processing."""
        bool_masks = []
        for param in param_list:
            if isinstance(param, pm.Mask) and param.bool_mask:
                bool_masks.append(param)
                param_list.remove(param)
        return bool_masks
    
    def assign_spatial_info(self, dataset):
        """
        Assigns the spatial information from the dataset to the class attributes.
        
        Args:
            dataset: The raster dataset to extract spatial information from.
        """
        self.crs = dataset.crs
        self.transform = dataset.transform
        print(f"Assigned CRS: {self.crs}, Transform: {self.transform}")