import os
import re
import json

import core.config as cfg
import core.utils as utils
import core.processing as pr
import numpy as np

class ProcessingPipeline:
    """
    A class to handle the complete processing pipeline dictated by a YAML file.
    
    Attributes:
    -----------
        yaml_file (str): The path to the YAML file containing the processing configuration.
    """
    def __init__(self, yaml_path=None):
        self.yaml_path = yaml_path
        self.config = cfg.config(self.yaml_path)
        self.crs = None
        self.transform = None
        self.mask = None
    
    def process_file(self):
        """
        Process the parameter or indicator based on the name.
        """
        # self.config.processes is a dict, so iterate over its items
        for process_name, process in self.config.processes.items():
            print(f"Processing {process_name}: {process["name"]}")

            self.dir_path = process["path"]

            param_list = self.init_parameters(process)

            processed_rasters = self.process_parameters(process, param_list)
            utils.show_raster(processed_rasters[0], title=f"{process_name} - Processed Raster 0")
            utils.show_raster(processed_rasters[1], title=f"{process_name} - Processed Raster 1")


            zonal_stats = self.calculate_stats(process, processed_rasters, param_list)

            self.process_vector(process, processed_rasters, zonal_stats)

    def init_parameters(self, process):
        """
        Initialize the parameters based on the process configuration.
        
        Args:
            process: The process configuration dictionary.
        
        Returns:
            List: A list of Parameter objects initialized with the raster data.
        """
        param_file_dicts = self.get_file_paths(self.get_param_names(process))
        mask_file_dicts = self.get_file_paths(self.get_mask_names(process))

        param_list = []
        for idx, (param_name, file_path) in enumerate(param_file_dicts.items()):
            param = Parameter(param_name, file_path)
            if idx == 0:
                self.assign_spatial_info(param.dataset)
                self.mask = param.coverage_mask()
            param_list.append(param)
        
        for mask_name, file_path in mask_file_dicts.items():
            mask_param = Parameter(mask_name, file_path)
            mask_param.mask = True
            param_list.append(mask_param)
        
        return param_list
    
    def process_parameters(self, process, param_list):
        """
        Process the raster data based on the configuration.

        Returns:
            List: A list of processed raster data
        """
        raster_list = self.threshold(process, param_list)

        # boolean filters
        for task in process["pipeline"]: 
            if "majority" in task["task"]:    
                print(f"Applying majority filter")
                iterations = self.get_task_param(task, "iterations")
                size = self.get_task_param(task, "size")

                iterations = 1 if iterations is None else iterations
                size = 3 if size is None else size 
                raster_list = pr.list_majority_filter(raster_list, iterations=iterations, size=size)

            elif "boundary" in task["task"]:
                print(f"Applying boundary clean filter ")
                # raster_list = pr.list_boundary_clean(raster_list, iterations=process["boundary_clean"]["iterations"], radius=process["boundary_clean"]["radius"])

            elif "sieve" in task["task"]:
                print(f"Applying sieve filter ")
                threshold = self.get_task_param(task, "threshold")
                iterations = self.get_task_param(task, "iterations")
                connectedness = self.get_task_param(task, "connectedness")
                
                threshold = 9 if threshold is None else threshold
                iterations = 1 if iterations is None else iterations
                connectedness = 4 if connectedness is None else connectedness
                
                raster_list = utils.list_sieve_filter(raster_list, iterations=iterations, threshold=threshold, crs=self.crs, transform=self.transform)

        for i in range(len(raster_list)):
            raster_list[i] = raster_list[i] * self.mask
        raster_list = list(raster_list)
        raster_list.insert(0, self.mask)
        return raster_list

    def threshold(self, process, param_list):
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
            if not isinstance(param, Parameter):
                raise TypeError(f"Expected Parameter object, got {type(param)}")
            
            # Apply median filter
            median_filter = param.median_filter(iterations=process["thresholds"]["median"]["iterations"], size=process["thresholds"]["median"]["size"])
            median_filter = param.raster
            if param.mask:
                masks_thresholded_list.append(param.threshold(median_filter, process["thresholds"]["masks"][param.name]))
            else:
                param_thresholded_list.append(param.threshold(median_filter, process["thresholds"]["parameters"][param.name]))
            
        # Combine the thresholded rasters
        if len(masks_thresholded_list) > 0 or len(param_thresholded_list) > 1:
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
            if not isinstance(param, Parameter):
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

    def get_param_names(self, process):
        """
        Get the parameter names from the process configuration.
        """
        return list(process["thresholds"]["parameters"].keys())

    def get_mask_names(self, process):
        """
        Get the file paths of the raster data based on the configuration.
        """
        # Implement the logic to get raster file paths
        if "masks" not in process["thresholds"] or process["thresholds"]["masks"] is None:
            print("No masks found in the process configuration.")
            return []
        return list(process["thresholds"]["masks"].keys())

    def get_file_paths(self, names):
        """
        Returns the file path of the parameter raster or paths for indicators.
        """
        files = os.listdir(self.dir_path)
        files_dict = {}

        for param in names:
            file_path = self._find_file(files, param)
            if file_path:
                files_dict[param] = file_path
            else:
                print(f"File for parameter {param} not found in {self.dir_path}")        

        return files_dict

    def _find_file(self, files, param):
        """
        Helper function to find the file for a given parameter in the directory.
        """
        pattern = re.compile(rf".*{param}.*\.IMG$")
        for f in files:
            match = pattern.match(f)
            if match:
                return os.path.join(self.dir_path, f)
        return None
    
    # def open_rasters(self, process):
    #     """
    #     Open a raster file and return the dataset. Also assigns spatial info to the class.
        
    #     Args:
    #         raster_path (str): The path to the raster file.
        
    #     Returns:
    #         List: The list of raster
    #     """
    #     param_names = self.get_param_names(process)
    #     param_path_dict = self.get_file_paths(param_names)
        
    #     raster_list = []
    #     for param, raster_path in param_path_dict.items():
    #         print(f"Opening raster for parameter {param} from {raster_path}")

    #         dataset = utils.open_raster(raster_path)
    #         raster = utils.open_raster_band(dataset, 1)
    #         raster_list.append(raster)

    #         if self.crs is None and self.transform is None:
    #             self.assign_spatial_info(dataset)
    #     return raster_list
    

class Parameter:
    def __init__(self, name: str, raster_path: str, defaults=None):
        self.name = name
        self.raster_path = raster_path
        self.dataset = utils.open_raster(raster_path)
        self.raster = utils.open_raster_band(self.dataset, 1)
        self.defaults = defaults if defaults else pr.Defaults()
        self._median_filter_result = None
        self.mask = False
    
    def median_filter(self, size=3, iterations=None):
        """Apply a median filter to the raster data."""
        if self._median_filter_result is None:
            num_median_filter = iterations if iterations else self.defaults.get_num_median_filter(self.name)
            self._median_filter_result = pr.median_kernel_filter(self.raster, iterations=num_median_filter, size=size)
        return self._median_filter_result

    def threshold(self, raster=None, thresholds=None):
        """Apply thresholds to the raster data and return a list."""
        if raster is None:
            raster = self.raster
        if thresholds is None:
            thresholds = self.defaults.get_thresholds(self.name)
        return pr.full_threshold(raster, thresholds)
    
    def majority_filter(self, raster_list, size=3, iterations=None):
        """Apply a majority filter to a list of raster data."""
        num_majority_filter = iterations if iterations else self.defaults.get_num_majority_filter(self.name)
        return pr.list_majority_filter(raster_list, size=size, iterations=num_majority_filter)

    def boundary_clean(self, raster_list, radius=1,  iterations=None):
        """Apply a boundary clean filter to a list of raster data."""
        num_boundary_clean = iterations if iterations else self.defaults.get_num_boundary_clean(self.name)
        return pr.list_boundary_clean(raster_list, iterations=num_boundary_clean, radius=radius)
    
    def sieve_filter(self, raster_list, threshold=9, iterations=3):
        """Apply a sieve filter to a list of raster data."""
        return utils.list_sieve_filter(raster_list, threshold=threshold, profile=self, iterations=iterations)
    
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
    
    def mask_list(self, raster_list):
        """Return a list of masked rasters based on the coverage mask."""
        coverage_mask = self.coverage_mask()
        for i in range(len(raster_list)):
            raster_list[i] = utils.mask(raster_list[i], coverage_mask)
        
        return raster_list

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
   
     Attributes :
    -----------
        path (str): The file path to the raster data.
        param (str): The parameter to be processed. 
    
    """
    def __init__(self, dir_path, output_path, name): 
        self.dir_path = dir_path
        self.output_path = output_path
        self.defaults = pr.Defaults()
        self.name = name

    def raster_process(self, param):
        """
        returns a list of processed rasters for the parameter.
        """
        # Apply median filter
        median_filter = param.median_filter()
        # utils.save_raster(median_filter, self.output_path, "prog_median_filter.tif", param.dataset.profile)

        # Apply threshold
        thresholded = param.threshold(median_filter)
        # utils.save_raster(thresholded[0], self.output_path, "prog_thresholded_0.tif", param.dataset.profile)

        # Apply majority filter
        majority_filtered = param.majority_filter(thresholded)
        # utils.save_raster(majority_filtered[0], self.output_path, "prog_majority_filtered_0.tif", param.dataset.profile)

        # Apply boundary clean filter
        boundary_cleaned = param.boundary_clean(majority_filtered)
        # utils.save_raster(boundary_cleaned[0], self.output_path, "prog_boundary_cleaned_0.tif", param.dataset.profile)

        # Second filters
        majority_filtered_3 = param.majority_filter(boundary_cleaned, iterations=3)
        # utils.save_raster(majority_filtered_3[0], self.output_path, "prog_majority_filtered_3_0.tif", param.dataset.profile)

        sieve_filtered = param.sieve_filter(majority_filtered_3)
        # utils.save_raster(sieve_filtered[0], self.output_path, "prog_sieve_filtered_0.tif", param.dataset.profile)

        # boundary_cleaned2 = param.boundary_clean(sieve_filtered)
        # # utils.show_raster((boundary_cleaned2[0]), title=f"{param.name} - boundary_cleaned2")

        final_rater_list = param.mask_list(sieve_filtered)
        # utils.save_raster(final_rater_list[0], self.output_path, "prog_final_rater_list_0.tif", param.dataset.profile)

        # utils.show_raster((final_rater_list[0]), title=f"{param.name} - Final Processed Raster")


        return final_rater_list

    def process_parameter(self):
        """
        Full processing of a parameter.
        """
        print(f"Processing parameter: {self.name}")

        param_file_dicts = self.get_param_file_paths()

        param = Parameter(self.name, param_file_dicts[self.name], self.defaults)
        
        rater_list = self.raster_process(param)
        
        # labeled raster
        labeled_raster_list = pr.list_label_clusters(rater_list)

        zonal_stats = pr.list_zonal_stats(labeled_raster_list, param)
        vector_list = pr.list_vectorize_dict(labeled_raster_list, zonal_stats, param)
        vector_stack = utils.merge_polygons(vector_list)

        # utils.save_shapefile(vector_stack, self.output_path, f"t1250_{self.name}_stack.shp")

    def process_indicator(self):
            """
            Full processing of an indicator.
            """
            print(f"Processing indicator: {self.name}")
            
            param_file_dicts = self.get_param_file_paths()
            mask_file_dicts = self.get_mask_file_paths()

            indicator_param_list = []
            for param_name, file_path in param_file_dicts.items():
                print(f"Processing parameter: {param_name} from file: {file_path}")
                
                param = Parameter(param_name, file_path, self.defaults)
                # Process the parameter
                indicator_param_list.append(self.raster_process(param))
            
            indicator_mask_list = []
            for mask_name, file_path in mask_file_dicts.items():
                print(f"Processing mask: {mask_name} from file: {file_path}")
                
                mask_param = Parameter(mask_name, file_path, self.defaults)
                # Process the mask
                indicator_mask_list.append(self.raster_process(mask_param))

            param_levels = list(zip(*indicator_param_list))

            # # Combine all masks with logical OR, then invert for valid mask
            combined_mask = np.logical_not(np.logical_or.reduce(indicator_mask_list)).astype(np.uint32)
            
            if combined_mask.ndim == 3 and combined_mask.shape[0] == 1:
                combined_mask = np.squeeze(combined_mask, axis=0)

            # utils.show_raster(combined_mask, title=f"{self.name} - combined_mask")

            raster_list = [
                np.prod(level_rasters, axis=0)
                for level_rasters in param_levels
            ]
            
            for i in range(len(raster_list)):
                if raster_list[i].ndim == 3 and raster_list[i].shape[0] == 1:
                    raster_list[i] = np.squeeze(raster_list[i], axis=0)
                raster_list[i] = raster_list[i] * combined_mask

            # utils.show_raster(combined_mask, title=f"{self.name} - combined_mask")
            
            # Open the first parameter in param_file_dicts as a Parameter object
            first_param_name = list(param_file_dicts.keys())[0]
            temp_param = Parameter(first_param_name, param_file_dicts[first_param_name], self.defaults)

            raster_list = utils.list_sieve_filter(raster_list, threshold=12, profile=temp_param)

            # utils.show_raster(np.squeeze(raster_list[0]), title=f"{self.name} - Low Level")
            # utils.show_raster(np.squeeze(raster_list[1]), title=f"{self.name} - Mid Level")
            # utils.show_raster(np.squeeze(raster_list[2]), title=f"{self.name} - Hi Level")
            # labeled raster
            labeled_raster_list = pr.list_label_clusters(raster_list)

            zonal_stats = {}
            for param_name, file_path in param_file_dicts.items():
                param = Parameter(param_name, file_path, self.defaults)
                zonal_stats[param_name] = pr.list_zonal_stats(labeled_raster_list, param)

            if pr.check_stats_dict(zonal_stats):
                # Restructure the zonal stats dictionary
                zonal_stats = pr.restructure_stats(zonal_stats)

            vector_list = pr.list_vectorize_dict(labeled_raster_list, zonal_stats, temp_param)
            vector_stack = utils.merge_polygons(vector_list)

            utils.save_shapefile(vector_stack, self.output_path, f"{self.name}_stack2.shp")

    def get_param_file_paths(self):
        """
        Returns the file path of the parameter raster or paths for indicators.
        """
        files = os.listdir(self.dir_path)
        files_dict = {}

        if self.defaults.indicator_check(self.name):
            indicator_param_names = self.defaults.get_indicator_param_names()
        else:
            indicator_param_names = [self.name]
        
        for param in indicator_param_names:
            file_path = self._find_file(files, param)
            if file_path:
                files_dict[param] = file_path
            else:
                print(f"File for parameter {param} not found in {self.dir_path}")        

        return files_dict
    
    def get_mask_file_paths(self):
        """
        Returns the file path of the mask raster for indicators.
        """
        mask_param_names = self.defaults.get_indicator_mask_names()
        files_dict = {}

        files = os.listdir(self.dir_path)

        for param in mask_param_names:
            file_path = self._find_file(files, param)
            if file_path:
                files_dict[param] = file_path
            else:
                print(f"File for parameter {param} not found in {self.dir_path}")        

        return files_dict

    def _find_file(self, files, param):
        """
        Helper function to find the file for a given parameter in the directory.
        """
        pattern = re.compile(rf".*{param}.*\.IMG$")
        for f in files:
            match = pattern.match(f)
            if match:
                return os.path.join(self.dir_path, f)
        return None
