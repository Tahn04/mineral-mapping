"""
Raster processing operations and utilities.
"""
import time
import numpy as np
import xarray as xr
from tqdm import tqdm
from .. import raster_ops as ro
from .. import parameter as pm


class RasterProcessor:
    """
    Handles all raster processing operations including thresholding,
    filtering, and masking.
    """
    
    def __init__(self, config):
        self.config = config
        self.indicator = False
    
    def threshold(self, param_list, mask=None):
        """
        Apply median filter, thresholds, and masks to the data.
        
        Args:
            param_list: List of Parameter objects
            mask: Optional mask to apply
            
        Returns:
            Processed raster data
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

            # Apply mask if provided
            if mask is not None:
                preprocessed = preprocessed.where(mask, np.nan)
                
            self._save_raster(preprocessed, param)
            final = time.time()
            print(f"Raster saved execution time: {final - mid:.2f} seconds")
        
            # Separate masks from parameters
            if param.mask:
                masks_thresholded_list.append(param.threshold(preprocessed, param.thresholds))
            else:
                param_thresholded_list.append(param.threshold(preprocessed, param.thresholds))

        return self._combine_thresholded_rasters(param_thresholded_list, masks_thresholded_list)
    
    def apply_processing_pipeline(self, raster_list):
        """
        Execute the processing pipeline (filters, morphological operations, etc.).
        
        Args:
            raster_list: List of raster data
            
        Returns:
            Processed raster list
        """
        show_rasters = False
        if show_rasters:
            ro.show_raster(raster_list[0], title="threshold- Processed Raster lowest")
            
        for task in self.config.get_pipeline() if self.config.get_pipeline() else []:
            task_name = task.get("task", "")
            
            if "majority" in task_name:
                raster_list = self._apply_majority_filter(raster_list, task, show_rasters, task_name)
            elif "boundary" in task_name:
                raster_list = self._apply_boundary_clean(raster_list, task, show_rasters, task_name)
            elif "sieve" in task_name:
                raster_list = self._apply_sieve_filter(raster_list, task, show_rasters, task_name)
            elif "open" in task_name:
                raster_list = self._apply_binary_opening(raster_list, task, show_rasters, task_name)
        
        return raster_list
    
    def clean_rasters(self, raster_list, param_list, full_mask):
        """
        Apply coverage mask to the raster list.
        
        Args:
            raster_list: List of raster data
            param_list: List of parameters
            full_mask: Combined coverage mask
            
        Returns:
            Cleaned raster list
        """
        if not raster_list:
            return []

        # Convert coverage_mask to numpy once if it's xarray
        if isinstance(full_mask, xr.DataArray):
            coverage_mask_np = full_mask.values
        else:
            coverage_mask_np = full_mask
        
        # Pre-allocate result list
        base_check = self.config.get_base_check()
        final_raster_list = []
        if base_check:
            final_raster_list.append(coverage_mask_np.astype(np.uint8))
        
        # Process rasters
        for raster in tqdm(raster_list, desc="Cleaning rasters"):
            if isinstance(raster, np.ndarray):
                cleaned = raster * coverage_mask_np
            else:
                # Handle xarray case
                if hasattr(raster, 'values'):
                    cleaned = raster.values * coverage_mask_np
                else:
                    cleaned = np.asarray(raster) * coverage_mask_np
            
            final_raster_list.append(cleaned)
        
        return final_raster_list
    
    def calculate_param_coverage_mask(self, param_list):
        """Calculate the coverage mask for the raster data."""
        if not param_list:
            return None
        
        if len(param_list) == 1:
            param = param_list[0]
            if isinstance(param, xr.DataArray):
                return ~np.isnan(param.values)
            else:
                return param.coverage_mask()
        
        # For multiple parameters, use efficient reduction
        masks = []
        for param in param_list:
            if isinstance(param, xr.DataArray):
                mask = ~np.isnan(param.values)
            else:
                mask = param.coverage_mask()
            masks.append(mask)
        
        return np.logical_and.reduce(masks)
    
    def complete_mask(self, param_list, post_processing_masks=None):
        """Calculate the complete coverage mask."""
        coverage_mask = self.calculate_param_coverage_mask(param_list)
   
        if post_processing_masks:
            for mask in post_processing_masks:
                val = mask.thresholds[0]
                coverage_mask = ro.clip_raster(coverage_mask, mask.dataset, val=val)
        return coverage_mask
    
    def get_post_processing_masks(self, param_list):
        """Get parameters that will be applied as masks after processing."""
        keep_shape_masks = []
        for param in param_list[:]:  # Iterate over copy
            if isinstance(param, pm.Mask) and param.keep_shape:
                keep_shape_masks.append(param)
                param_list.remove(param)
        return param_list, keep_shape_masks
    
    # Private methods
    def _save_raster(self, data, param):
        """Save raster data if stats are to be computed."""
        if self.config.get_stats():
            from .. import file_handler as fh
            preprocessed_path = fh.FileHandler().create_temp_file("preprocessed", "tif")
            ro.save_raster_fast_rasterio(data, param.crs, param.transform, preprocessed_path)
            param.preprocessed_path = preprocessed_path
    
    def _combine_thresholded_rasters(self, param_thresholded, masks_thresholded=[]):
        """Combine thresholded rasters with detailed metadata handling."""
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
                combined_mask_raw = masks_thresholded[0].copy() 
                for mask in masks_thresholded[1:]:
                    combined_mask_raw = xr.ufuncs.logical_or(combined_mask_raw, mask[0])
                combined_mask = xr.ufuncs.logical_not(combined_mask_raw)
            else:
                combined_mask = xr.ones_like(template, dtype='uint32')
            
            # Process parameters (multiplicative)
            if len(param_thresholded) > 1:
                param_thresholded = self._assign_thresholds_to_params(param_thresholded)
                combined_params = param_thresholded[0].copy()
                for param in param_thresholded[1:]:
                    combined_params = combined_params * param
            else:
                combined_params = param_thresholded[0]
            
            # Apply mask
            result = combined_params * combined_mask[0]
            result.attrs.update({
                'operation': 'combined_thresholded_rasters',
                'num_masks': len(masks_thresholded),
                'num_params': len(param_thresholded)
            })
            
            return result
        else:
            return param_thresholded[0]
    
    def _assign_thresholds_to_params(self, param_thresholded):
        """Assign threshold coordinates to parameters."""
        num_thresholds = len(param_thresholded[0].threshold)
        if num_thresholds > 0:
            threshold_list = list(range(1, num_thresholds + 1))
            new_param_thresholded = []
            for param in param_thresholded:
                param = param.assign_coords(threshold=threshold_list)
                new_param_thresholded.append(param)
        return new_param_thresholded
    
    def _apply_majority_filter(self, raster_list, task, show_rasters, task_name):
        """Apply majority filter to raster list."""
        iterations = self.config.get_task_param(task, "iterations") or 1
        size = self.config.get_task_param(task, "size") or 3
        raster_list = ro.dask_list_majority_filter(raster_list, iterations=iterations, size=size)
        if show_rasters:
            ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")
        return raster_list
    
    def _apply_boundary_clean(self, raster_list, task, show_rasters, task_name):
        """Apply boundary cleaning to raster list."""
        iterations = self.config.get_task_param(task, "iterations") or 1
        size = self.config.get_task_param(task, "size") or 3
        raster_list = ro.list_boundary_clean(raster_list, iterations=iterations, radius=size)
        if show_rasters:
            ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")
        return raster_list
    
    def _apply_sieve_filter(self, raster_list, task, show_rasters, task_name):
        """Apply sieve filter to raster list."""
        threshold = self.config.get_task_param(task, "threshold") or 9
        iterations = self.config.get_task_param(task, "iterations") or 1
        connectedness = self.config.get_task_param(task, "connectedness") or 4
        
        raster_list = ro.list_sieve_filter_rio(
            raster_list,
            iterations=iterations,
            threshold=threshold,
            connectedness=connectedness
        )
        if show_rasters:
            ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")
        return raster_list
    
    def _apply_binary_opening(self, raster_list, task, show_rasters, task_name):
        """Apply binary opening to raster list."""
        iterations = self.config.get_task_param(task, "iterations") or 1
        size = self.config.get_task_param(task, "size") or 3
        raster_list = ro.list_binary_opening(raster_list, iterations=iterations, size=size)
        if show_rasters:
            ro.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")
        return raster_list
