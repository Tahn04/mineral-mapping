import yaml
from pydantic import BaseModel
import os
import core.parameter as pm
import re

class Config:
    """
    Configuration handler for the mineral mapping application.
    Loads and provides access to settings from a YAML file.
    """
    def __init__(self, yaml_file=None, process=None):
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/config.yaml"))
        self.yaml = True
        if yaml_file is None:
            self.yaml = False
        self.yaml_file = yaml_file or default_path
        self._config = None
        self.curr_process = None
        self.process = process or "default" 
        self.params = []
        self.load_config()

    def get_parameters_list(self):
        """
        Initialize the parameters based on the process configuration.
        
        Returns:
            List: A list of Parameter objects initialized with the raster data.
        """
        return self.params
        
    def config_array(self, param, crs, transform, mask=None):
        """
        Initialize the configuration with an array and its metadata.
        
        Args:
            array: The raster data as a numpy array.
            crs: Coordinate Reference System of the raster data.
            transform: Affine transformation for the raster data.
        """
        for key, value in param.items():
            if isinstance(value, tuple) and len(value) == 2:
                # If the value is a list, assume it's a list of arrays
                param = pm.Parameter(
                    key, 
                    array=value[0], 
                    crs=crs, 
                    transform=transform, 
                    thresholds=value[1] if len(value) > 1 else None
                )
                self.params.append(param)
            else:
                raise ValueError("Provide thresholds")
        if mask is not None:
            for key, value in mask.items():
                if isinstance(value, list):
                    # If the value is a list, assume it's a list of arrays
                    mask_param = pm.Parameter(
                        key, 
                        array=value[0], 
                        crs=crs, 
                        transform=transform, 
                        thresholds=value[1] if len(value) > 1 else None
                    )
                    mask_param.mask = True
                    self.params.append(mask_param)
                else:
                    raise ValueError("Provide thresholds for mask")

    def config_files(self, rast, mask=None):
        """
        Initialize the configuration with file paths for parameters and masks.
        
        Args:
            param: Dictionary of parameter names and their file paths.
            mask: Dictionary of mask names and their file paths.
        """

        # for key, value in rast.items():
        #     if isinstance(value, tuple) and len(value) == 2:
        #         # If the value is a tuple, assume it's a file path and thresholds
        #         param_file_dicts[key] = (value[0], value[1])

        self.init_parameters(rast, mask)

    def config_yaml(self):
        """
        Initialize the configuration from a YAML file.
        
        Args:
            yaml_file (str): Path to the YAML configuration file.
            process (str): Name of the process to set as current.
        """
        
        param_file_dicts = self.get_nested('processes', self.process, 'thresholds', 'parameters', default={})
        mask_file_dicts = self.get_nested('processes', self.process, 'thresholds', 'masks', default={})
        
        self.init_parameters(param_file_dicts, mask_file_dicts)

    def init_parameters(self, param_file_dicts, mask_file_dicts):
        """
        Initialize the parameters based on the process configuration.
        
        Args:
            process: The process configuration dictionary.
        
        Returns:
            List: A list of Parameter objects initialized with the raster data.
        """
        # param_file_dicts = self.get_file_paths(self.get_param_names())
        # mask_file_dicts = self.get_file_paths(self.get_mask_names())

        param_list = []
        for param_name, parameters in param_file_dicts.items():
            param = pm.Parameter(name=param_name, raster_path=parameters[0], thresholds=parameters[1] if len(parameters) > 1 else None)
            param_list.append(param)
        
        if mask_file_dicts is not None:
            for mask_name, parameters in mask_file_dicts.items():
                mask_param = pm.Parameter(mask_name, raster_path=parameters[0], thresholds=parameters[1] if len(parameters) > 1 else None)
                mask_param.mask = True
                param_list.append(mask_param)
        
        self.params = param_list

    def get_file_paths(self, names):
        """
        Returns the file path of the parameter raster or paths for indicators.
        """
        files = os.listdir(self.get_dir_path())
        files_dict = {}

        for param in names:
            file_path = self._find_file(files, param)
            if file_path:
                files_dict[param] = file_path
            else:
                print(f"File for parameter {param} not found in {self.get_dir_path()}")        

        return files_dict

    def _find_file(self, files, param):
        """
        Helper function to find the file for a given parameter in the directory.
        """
        pattern = re.compile(rf".*{param}.*\.IMG$")
        for f in files:
            match = pattern.match(f)
            if match:
                return os.path.join(self.get_dir_path(), f)
        return None


    def load_config(self):
        """Load the configuration from the YAML file."""
        if self._config is None:
            with open(self.yaml_file, 'r') as file:
                self._config = yaml.safe_load(file)
        # Set top-level keys as attributes for convenience
        if self.process:
            self.set_current_process(self.process)
        #     configuration = self._config.get(self.process, {})
        # else:
        #     configuration = self._config
        # for key, value in configuration.items():
        #     setattr(self, key, value)

    def get(self, key, default=None):
        """Get a top-level config value by key."""
        return self._config.get(key, default)

    def get_nested(self, *keys, default=None):
        """
        Get a nested config value by a sequence of keys.
        Example: config.get_nested('processes', 'my_process', 'thresholds')
        """
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_processes(self):
        """Return the processes dictionary from the config."""
        return self._config.get('processes', {})
    
    def get_current_process(self):
        """Get the current process configuration."""
        if self.curr_process is None:
            raise ValueError("Current process is not set.")
        processes = self.get_processes()
        if self.curr_process not in processes:
            raise ValueError(f"Process '{self.curr_process}' not found in configuration.")
        return processes[self.curr_process]

    def get_median_config(self):
        """Get the median configuration from the config."""
        if self.curr_process is None:
            raise ValueError("Current process is not set.")
        
        return self.get_nested('processes', self.curr_process, 'thresholds', 'median', default={})

    def get_masks(self):
        """Get mask names for the current process."""
        if self.curr_process is None:
            raise ValueError("Current process is not set.")
        return self.get_nested('processes', self.curr_process, 'thresholds', 'masks', default={})

    def get_pipeline(self):
        """Get the pipeline steps for the current process."""
        if self.curr_process is None:
            raise ValueError("Current process is not set.")
        return self.get_nested('processes', self.curr_process, 'pipeline', default=[])
    
    def get_dir_path(self):
        """Get the directory path for the current process."""
        process = self.get_current_process()
        return process.get("path", "") 
    
    def get_param_names(self):
        """
        Get the parameter names from the current process configuration.
        """
        process = self.get_current_process()
        return list(process["thresholds"]["parameters"].keys())

    def get_mask_names(self):
        """
        Get the mask names from the current process configuration.
        """
        process = self.get_current_process()
        if "masks" not in process["thresholds"] or process["thresholds"]["masks"] is None:
            print("No masks found in the process configuration.")
            return []
        return list(process["thresholds"]["masks"].keys())
    
    def get_task_param(self, task, param_name):
        """
        Get a specific parameter for a task in the current process pipeline.
        """
        if param_name in task:
            return task.get(param_name)
        else:
            return None
        
    def get_output_path(self):
        """Get the output path for the current process."""
        process = self.get_current_process()
        return process['vectorization'].get('output_dict', '')
    
    def get_driver(self):
        """Get the driver for the current process."""
        process = self.get_current_process()
        return process['vectorization'].get('driver', 'pandas')
    
    def get_cs(self):
        """Get the coordinate reference system for the current process."""
        process = self.get_current_process()
        cs = process['vectorization'].get('cs', None)
        # mars_gcs = "GEOGCS[\"GCS_Mars_2000\",DATUM[\"D_Mars_2000\",SPHEROID[\"Mars_2000_IAU_IAG\",3396190,169.894447223612]],PRIMEM[\"Reference_Meridian\",0],UNIT[\"Degree\",0.0174532925199433]]"
        if cs is None or cs == "GCS":
            gcs = "GEOGCS[" + self.crs.split("GEOGCS[")[1].split("]")[0] + "]"
            return gcs
        elif cs == "PCS":
            return self.crs
        else:
            return cs

    # Setters
    def set_current_process(self, process_name):
        """Set the current process name."""
        if process_name in self.get_processes():
            self.curr_process = process_name
        else:
            raise ValueError(f"Process '{process_name}' not found in configuration.")
    