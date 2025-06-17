import os
import re
import json

import core.config as cfg
import core.utils as utils
import core.processing as pr
import core.parameter as pm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class ProcessingPipeline:
    """
    A class to handle the complete processing pipeline dictated by a YAML file.
    
    Attributes:
    -----------
        yaml_file (str): The path to the YAML file containing the processing configuration.
    """
    def __init__(self, config):
        # self.yaml_path = yaml_path
        # self.config = cfg.Config(self.yaml_path)
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
            # self.config.set_current_process(process_name)
            
            param_list = self.config.get_parameters_list()

            processed_rasters = self.process_parameters(param_list)
            # utils.show_raster(processed_rasters[1], title=f"{process_name} - Processed Raster 1")
            # utils.show_raster(processed_rasters[-1], title=f"{process_name} - Processed Raster 10")

            self.vectorize(processed_rasters, param_list)

            # zonal_stats = self.calculate_stats(process, processed_rasters, param_list)

            # self.process_vector(process, processed_rasters, zonal_stats)
    
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
        driver = self.config.get_driver()
        thresholds = self.assign_thresholds(raster_list, param_list)
        polygons = pr.list_vectorize(raster_list, thresholds, self.crs, self.transform)

        vector_stack = pr.list_zonal_stats2(polygons, param_list, self.crs, self.transform)

        vector_stack = self.assign_color(vector_stack)
        if driver == "pandas":
            return vector_stack

        # mars_gcs = {
        #     "proj": "longlat",
        #     "a": 3396190,
        #     "rf": 169.894447223612,
        #     "no_defs": True
        # }
        crs_wkt = self.crs.to_wkt() 
        test_gcs = crs_wkt[0]

        mars_gcs = "GEOGCS[\"GCS_Mars_2000\",DATUM[\"D_Mars_2000\",SPHEROID[\"Mars_2000_IAU_IAG\",3396190,169.894447223612]],PRIMEM[\"Reference_Meridian\",0],UNIT[\"Degree\",0.0174532925199433]]"

        # Reproject to GCS
        gdf_gcs = vector_stack.to_crs(mars_gcs)

        output_dict = self.config.get_output_path()
        process_name = self.config.get_current_process()["name"]
        utils.save_shapefile(gdf_gcs, output_dict, f"{process_name}_final.geojson", driver=driver)

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
            utils.show_raster(raster_list[0], title="threshold- Processed Raster lowest")
            # utils.save_raster(raster_list[0], r"\\lasp-store\home\taja6898\Documents\Mars_Data\T1250_demo_parameters", "MC13_thresholded_0.tif", param_list[0].dataset.profile)
        # boolean filters 
        for task in self.config.get_pipeline():
            task_name = task.get("task", "")
            if "majority" in task_name:
                iterations = self.config.get_task_param(task, "iterations")
                size = self.config.get_task_param(task, "size")    

                iterations = 1 if iterations is None else iterations
                size = 3 if size is None else size
                raster_list = pr.list_majority_filter(raster_list, iterations=iterations, size=size)
                if show_rasters:
                    utils.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")

            elif "boundary" in task_name:
                iterations = self.config.get_task_param(task, "iterations")
                radius = self.config.get_task_param(task, "radius")

                iterations = 1 if iterations is None else iterations
                radius = 1 if radius is None else radius
                raster_list = pr.list_boundary_clean(raster_list, iterations=iterations, radius=radius)
                if show_rasters:
                    utils.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")

            elif "sieve" in task_name:
                threshold = self.config.get_task_param(task, "threshold")
                iterations = self.config.get_task_param(task, "iterations")
                connectedness = self.config.get_task_param(task, "connectedness")

                threshold = 9 if threshold is None else threshold
                iterations = 1 if iterations is None else iterations
                connectedness = 4 if connectedness is None else connectedness

                raster_list = utils.list_sieve_filter(
                    raster_list,
                    iterations=iterations,
                    threshold=threshold,
                    crs=self.crs,
                    transform=self.transform,
                    connectedness=connectedness
                )
                if show_rasters:
                    utils.show_raster(raster_list[0], title=f"{task_name} - Processed Raster lowest")

        mask = target_param.coverage_mask()
        for i in range(len(raster_list)):
            raster_list[i] = raster_list[i] * mask
        raster_list = list(raster_list)
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

            # preproccessing = param.median_filter(iterations=median_iterations, size=median_size)
            # utils.show_raster(preproccessing, title="median_filter")

            preproccessing = param.new_median_filter(iterations=median_iterations, size=median_size)
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

            raster_list = [
                np.prod(level_rasters, axis=0)
                for level_rasters in param_levels
            ]
            
            for i in range(len(raster_list)):
                if raster_list[i].ndim == 3 and raster_list[i].shape[0] == 1: # Check if alwasy false
                    raster_list[i] = np.squeeze(raster_list[i], axis=0)
                raster_list[i] = raster_list[i] * combined_mask

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
            thresholds = [i for i in range(size)]
        else:
            thresholds = param_list[0].get_thresholds() # only one parameter in this case
            thresholds.insert(0, 0)  # Insert a zero threshold for the mask
        
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

        thresholds = gdf['value'].unique()
        cmap = plt.get_cmap(colormap, len(thresholds))
        color_map = {val: mcolors.to_hex(cmap(i)) for i, val in enumerate(sorted(thresholds))}

        gdf['hex_color'] = gdf['value'].map(color_map)
        return gdf
    
    def get_task_param(self, task, parameter):
        """
        Get the parameters for a specific task from the process configuration.
        
        Args:
            task: The task configuration dictionary.
        
        Returns:
            Dict: A dictionary of parameters for the task.
        """
        if parameter in task:
            return task[parameter]
        return None
    
    def assign_spatial_info(self, dataset):
        """
        Assigns the spatial information from the dataset to the class attributes.
        
        Args:
            dataset: The raster dataset to extract spatial information from.
        """
        self.crs = dataset.crs
        self.transform = dataset.transform
        print(f"Assigned CRS: {self.crs}, Transform: {self.transform}")
    
    def calculate_stats(self, process, raster_list, param_list):
        """
        Calculate statistics for the raster data based on the configuration.
        """
        labeled_raster_list = pr.list_label_clusters(raster_list)

        zonal_stats = {}
        for param in param_list:
            if not isinstance(param, pm.Parameter):
                raise TypeError(f"Expected Parameter object, got {type(param)}")
            # if param.mask:
            #     continue
            zonal_stats[param.name] = pr.list_zonal_stats(labeled_raster_list, param)

        if pr.check_stats_dict(zonal_stats):
            # Restructure the zonal stats dictionary
            zonal_stats = pr.restructure_stats(zonal_stats)

        return zonal_stats

    def process_vector(self, process, raster_list, zonal_stats):
        """
        Process the  vector data based on the configuration.
        """
        labeled_raster_list = pr.list_label_clusters(raster_list)

        vector_list = pr.list_vectorize_dict(labeled_raster_list, zonal_stats, crs=self.crs, transform=self.transform)
        vector_stack = utils.merge_polygons(vector_list)

        utils.save_shapefile(vector_stack, process["vectorization"]["output_dict"], f"{process["name"]}_new_code2.shp")

    # def init_parameters(self):
    #     """
    #     Initialize the parameters based on the process configuration.
        
    #     Args:
    #         process: The process configuration dictionary.
        
    #     Returns:
    #         List: A list of Parameter objects initialized with the raster data.
    #     """
    #     param_file_dicts = self.get_file_paths(self.config.get_param_names())
    #     mask_file_dicts = self.get_file_paths(self.config.get_mask_names())

    #     param_list = []
    #     for idx, (param_name, file_path) in enumerate(param_file_dicts.items()):
    #         param = pm.Parameter(param_name, file_path)
    #         if idx == 0:
    #             self.assign_spatial_info(param.dataset)
    #             self.mask = param.coverage_mask()
    #         param_list.append(param)
        
    #     for mask_name, file_path in mask_file_dicts.items():
    #         mask_param = pm.Parameter(mask_name, file_path)
    #         mask_param.mask = True
    #         param_list.append(mask_param)
        
    #     return param_list

    # def get_file_paths(self, names):
    #     """
    #     Returns the file path of the parameter raster or paths for indicators.
    #     """
    #     files = os.listdir(self.config.get_dir_path())
    #     files_dict = {}

    #     for param in names:
    #         file_path = self._find_file(files, param)
    #         if file_path:
    #             files_dict[param] = file_path
    #         else:
    #             print(f"File for parameter {param} not found in {self.dir_path}")        

    #     return files_dict

    # def _find_file(self, files, param):
    #     """
    #     Helper function to find the file for a given parameter in the directory.
    #     """
    #     pattern = re.compile(rf".*{param}.*\.IMG$")
    #     for f in files:
    #         match = pattern.match(f)
    #         if match:
    #             return os.path.join(self.config.get_dir_path(), f)
    #     return None
