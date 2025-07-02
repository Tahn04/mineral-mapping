import os
import re
import json

import core.config as cfg
import core.raster_ops as ro
import core.vector_ops as vo
import core.parameter as pm
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shutil

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
        self.indication = False
    
    def process_file(self):
        """
        Process the parameter or indicator based on the name.
        """
        # for process_name, process in tqdm(self.config.processes.items(), desc="Processing Processes"):
        process = self.config.get_current_process()
        print()
        for _ in tqdm(range(1), desc=f"Processing: {process["name"]}"):     
            param_list = self.config.get_parameters_list()
            processed_rasters = self.process_parameters(param_list)
            return self.vectorize(processed_rasters, param_list)

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

        # stats_dict = vo.list_raster_stats(band_array=param_list[0].raster, raster_list=raster_list, stats=["count", "mean", "std", "min", "max", 'median', '75p'])

        driver = self.config.get_driver()
        thresholds = self.assign_thresholds(raster_list, param_list)
        stats_list = self.config.get_stats()

        # # Old vectorization
        # start_old = time.time()
        # polygons = vo.list_vectorize(raster_list, thresholds, self.crs, self.transform, simplification_level)
        # vector_stack = vo.list_zonal_stats(polygons, param_list, self.crs, self.transform, stats_list)
        # end_old = time.time()
        # print(f"Old vectorization took {end_old - start_old:.2f} seconds")

        # New files based vector/stats
        start_new = time.time()
        gdf = vo.list_raster_to_shape_gdal(raster_list, thresholds, self.crs, self.transform, param_list, stats_list, simplification_level)
        end_new = time.time()
        print(f"New file-based vectorization took {end_new - start_new:.2f} seconds")

        colormap = self.config.get_colormap()
        if colormap:
            gdf = self.assign_color(gdf, colormap=colormap)

        # mars_gcs = {
        #     "proj": "longlat",
        #     "a": 3396190,
        #     "rf": 169.894447223612,
        #     "no_defs": True
        # }
        gdf.set_crs(self.crs, inplace=True)
        cs = self.config.get_cs(self.crs)
        projected_gdf = gdf.to_crs(cs) 
        
        if driver == "pandas":
            return projected_gdf
        
        output_dict = self.config.get_output_path()
        filename = self.config.get_output_filename()
        vo.save_shapefile(projected_gdf, output_dict, filename, driver=driver)
        return None

    def process_parameters(self, param_list):
        """
        Process the raster data based on the configuration.

        Returns:
            List: A list of processed raster data
        """
        raster_list = self.threshold(param_list)

        target_param = param_list[0]
        self.crs = target_param.get_crs()
        self.transform = target_param.get_transform()

        show_rasters = False
        if show_rasters:
            ro.show_raster(raster_list[0], title="threshold- Processed Raster lowest")
            # utils.save_raster(raster_list[0], r"\\lasp-store\home\taja6898\Documents\Mars_Data\T1250_demo_parameters", "MC13_thresholded_0.tif", param_list[0].dataset.profile)
        # boolean filters 
        for task in self.config.get_pipeline():
            task_name = task.get("task", "")
            if "majority" in task_name:
                iterations = self.config.get_task_param(task, "iterations")
                size = self.config.get_task_param(task, "size")    

                iterations = 1 if iterations is None else iterations
                size = 3 if size is None else size
                raster_list = ro.list_majority_filter(raster_list, iterations=iterations, size=size)
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

                raster_list = ro.list_sieve_filter(
                    raster_list,
                    iterations=iterations,
                    threshold=threshold,
                    crs=self.crs,
                    transform=self.transform,
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

        mask = target_param.coverage_mask()
        for i in range(len(raster_list)):
            raster_list[i] = raster_list[i] * mask
        raster_list = list(raster_list)

        base_check = self.config.get_base_check()
        if base_check:
            raster_list.insert(0, mask.astype(np.uint8))
        return raster_list

    def threshold(self, param_list):
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
            if not isinstance(param, pm.Parameter):
                raise TypeError(f"Expected Parameter object, got {type(param)}")
            
            # Apply median filter
            median_iterations = self.config.get_median_config().get("iterations", 0)
            median_size = self.config.get_median_config().get("size", 3)

            # preprocessing = param.median_filter(iterations=median_iterations, size=median_size)
            # utils.show_raster(preprocessing, title="median_filter")

            preproccessing = param.median_filter(iterations=median_iterations, size=median_size)
            # utils.show_raster(test_median, title="new_median_filter")
            # utils.save_raster(median_filter, r"\\lasp-store\home\taja6898\Documents\Code\mineral-mapping\outputs", f"T1250_median_filter_D2300.tif", param.dataset.profile)

            if param.mask:
                masks_thresholded_list.append(param.threshold(preproccessing, param.get_thresholds()))
            else:
                param_thresholded_list.append(param.threshold(preproccessing, param.get_thresholds()))

        # Combine the thresholded rasters
        if len(masks_thresholded_list) > 0 or len(param_thresholded_list) > 1:
            self.indication = True
            param_levels = list(zip(*param_thresholded_list))

            combined_mask = np.logical_not(np.logical_or.reduce(masks_thresholded_list)).astype(np.uint32)
            
            if combined_mask.ndim == 3 and combined_mask.shape[0] == 1: # Check if alwasy true
                combined_mask = np.squeeze(combined_mask, axis=0)
            # ro.show_raster(combined_mask, title="mask")
            raster_list = [
                np.prod(level_rasters, axis=0)
                for level_rasters in param_levels
            ]
            # ro.show_raster(raster_list[0], title="threshold - Processed Raster lowest")
            for i in range(len(raster_list)):
                if raster_list[i].ndim == 3 and raster_list[i].shape[0] == 1: # Check if alwasy false
                    raster_list[i] = np.squeeze(raster_list[i], axis=0)
                raster_list[i] = raster_list[i] * combined_mask
            # ro.show_raster(raster_list[0], title="threshold - Processed Raster lowest")
            return raster_list
        
        return param_thresholded_list[0]
    
    def assign_thresholds(self, raster_list, param_list):
        """
        Assign thresholds to the raster data based on the parameters.
        
        Args:
            raster_list: A list of processed raster data.
            param_list: A list of Parameter objects initialized with the raster data.
        
        Returns:
            List: A list of thresholds for each parameter.
        """
        if self.indication:
            size = len(raster_list)
            thresholds = [i + 1 for i in range(size)]
        else:
            thresholds = param_list[0].get_thresholds()
        
        base_check = self.config.get_base_check()
        if base_check:
            thresholds.insert(0, 0)
        
        return thresholds

    def assign_color(self, gdf, colormap="viridis"):
        """
        Assign colors to the geometries in the GeoDataFrame based on the thresholds.

        Args:
            gdf: The GeoDataFrame containing the geometries.
            colormap (str): The name of the matplotlib colormap to use for coloring the geometries.

        Returns:
            GeoDataFrame: The input GeoDataFrame with an added 'color' column.
        """

        thresholds = gdf['Threshold'].unique()
        cmap = plt.get_cmap(colormap, len(thresholds))
        color_map = {val: mcolors.to_hex(cmap(i)) for i, val in enumerate(sorted(thresholds))}

        gdf['hex_color'] = gdf['Threshold'].map(color_map)
        return gdf
    
    def assign_spatial_info(self, dataset):
        """
        Assigns the spatial information from the dataset to the class attributes.
        
        Args:
            dataset: The raster dataset to extract spatial information from.
        """
        self.crs = dataset.crs
        self.transform = dataset.transform
        print(f"Assigned CRS: {self.crs}, Transform: {self.transform}")