import yaml
import os
from .. import parameter as pm
from .. import file_handler as fh
import re
from pyproj import CRS, Transformer
from tqdm import tqdm
from typing import Dict, List, Union, Optional, Any

# Import the manager classes
from .parameter_manager import ParameterManager
from .process_manager import ProcessManager
from .output_manager import OutputManager
from .file_utilities import FileUtilities


class Config:
    """
    Configuration handler for the mineral mapping application.
    Loads and provides access to settings from a YAML file.
    
    This class now uses specialized managers for different aspects of configuration:
    - ParameterManager: Handles parameter and mask initialization
    - ProcessManager: Manages process-specific settings
    - OutputManager: Handles output paths, drivers, and vectorization settings
    """
    def __init__(self, yaml_file=None, process=None):
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/config.yaml"))
        self.yaml = True
        if yaml_file is None:
            self.yaml = False
        self.yaml_file = yaml_file or default_path
        self._config = None
        self.process = process or "default" 
        
        # Initialize managers
        self.parameter_manager = ParameterManager(self)
        self.process_manager = ProcessManager(self)
        self.output_manager = OutputManager(self)
        self.file_utilities = FileUtilities(self)
        
        self.load_config()

    # Delegate parameter-related methods to ParameterManager
    def get_parameters_list(self):
        """Get the list of initialized parameters."""
        return self.parameter_manager.get_parameters_list()

    def add_parameter(self, array, thresholds=None, crs=None, transform=None, name=None, median_iterations=1, median_size=3):
        """Add a new parameter to the configuration."""
        # Type validation
        if not hasattr(array, 'shape'):
            raise ValueError("'array' must be a numpy array or array-like object")
        
        if thresholds is not None and not isinstance(thresholds, (list, tuple)):
            raise ValueError("'thresholds' must be a list or tuple")
        
        if name is not None and not isinstance(name, str):
            raise ValueError("'name' must be a string")
        
        if not isinstance(median_iterations, int):
            raise ValueError("'median_iterations' must be an integer")

        if not isinstance(median_size, int):
            raise ValueError("'median_size' must be an integer")

        return self.parameter_manager.add_parameter(array, thresholds, crs, transform, name, median_iterations, median_size)

    def add_mask(self, array=None, crs=None, transform=None, name=None, threshold=None):
        """Add a new mask to the configuration."""
        # Type validation
        if array is not None and not hasattr(array, 'shape'):
            raise ValueError("'array' must be a numpy array or array-like object")
        
        if name is not None and not isinstance(name, str):
            raise ValueError("'name' must be a string")
        
        if threshold is not None and not isinstance(threshold, (int, float)):
            raise ValueError("'threshold' must be a number")
        
        return self.parameter_manager.add_mask(array, crs, transform, name, threshold)

    def config_array(self, param, crs, transform, mask=None):
        """Initialize the configuration with an array and its metadata."""
        return self.parameter_manager.config_array(param, crs, transform, mask)

    def name_check(self, name):
        """Check if the name is valid for a parameter or mask."""
        return self.parameter_manager._check_name(name)

    # Delegate process-related methods to ProcessManager
    def set_current_process(self, process_name):
        """Set the current process name."""
        # Type validation
        if not isinstance(process_name, str):
            raise ValueError("'process_name' must be a string")
        
        return self.process_manager.set_current_process(process_name)

    def get_processes(self):
        """Return the processes dictionary from the config."""
        return self.process_manager.get_processes()
    
    def get_current_process(self):
        """Get the current process configuration."""
        return self.process_manager.get_current_process()

    def get_nested(self, *keys, default=None):
        """Get a nested config value by a sequence of keys."""
        return self.process_manager.get_nested(*keys, default=default)

    def get_median_config(self):
        """Get the median configuration from the config."""
        return self.process_manager.get_median_config()

    def get_masks(self):
        """Get mask names for the current process."""
        return self.process_manager.get_masks()

    def get_pipeline(self):
        """Get the pipeline steps for the current process."""
        return self.process_manager.get_pipeline()
    
    def get_dir_path(self):
        """Get the directory path for the current process."""
        return self.process_manager.get_dir_path()
    
    def get_param_names(self):
        """Get the parameter names from the current process configuration."""
        return self.process_manager.get_param_names()

    def get_mask_names(self):
        """Get the mask names from the current process configuration."""
        return self.process_manager.get_mask_names()
    
    def get_task_param(self, task, param_name):
        """Get a specific parameter for a task in the current process pipeline."""
        return self.process_manager.get_task_param(task, param_name)

    # Delegate output-related methods to OutputManager
    def get_output_path(self):
        """Get the output path for the current process."""
        return self.output_manager.get_output_path()

    def get_driver(self):
        """Get the driver for the current process."""
        return self.output_manager.get_driver()
    
    def create_output_filename(self):
        """Get the output filename for the current process."""
        return self.output_manager.create_output_filename()
    
    def get_output_filename(self):
        """Get the output filename for the current process."""
        return self.output_manager.get_output_filename()

    def get_cs(self, crs):
        """Get the coordinate reference system for the current process."""
        return self.output_manager.get_cs(crs)

    def get_color(self):
        """Get the color for the current process."""
        return self.output_manager.get_color()

    def get_stats(self):
        """Get the statistics configuration for the current process."""
        return self.output_manager.get_stats()
    
    def get_base_check(self):
        """Check if the current process is set to run in base mode."""
        return self.output_manager.get_base_check()

    def get_base_stats(self):
        """Get the base statistics for the current process."""
        return self.output_manager.get_base_stats()
    
    def get_simplification_level(self):
        """Get the simplification level for vectorization."""
        return self.output_manager.get_simplification_level()
    
    def get_stack(self):
        """Check if the current process is set to stack results."""
        return self.output_manager.get_stack()

    # Delegate file-related methods to FileUtilities
    def get_file_paths(self, names):
        """Returns the file path of the parameter raster or paths for indicators."""
        return self.file_utilities.get_file_paths(names)

    def _find_file(self, files, param):
        """Helper function to find the file for a given parameter in the directory."""
        return self.file_utilities._find_file(files, param, self.get_dir_path())

    def config_files(self, rast, mask=None):
        """
        Initialize the configuration with file paths for parameters and masks.
        
        Args:
            rast: Dictionary of parameter names and their file paths
            mask: Dictionary of mask names and their file paths
        """
        # Type validation
        if not isinstance(rast, dict):
            raise ValueError("'rast' must be a dictionary")
        
        if mask is not None and not isinstance(mask, dict):
            raise ValueError("'mask' must be a dictionary or None")
        
        # Validate that all values are strings (file paths)
        for param_name, file_path in rast.items():
            if not isinstance(param_name, str):
                raise ValueError(f"Parameter name '{param_name}' must be a string")
            if not isinstance(file_path, str):
                raise ValueError(f"File path for parameter '{param_name}' must be a string")
        
        if mask:
            for mask_name, file_path in mask.items():
                if not isinstance(mask_name, str):
                    raise ValueError(f"Mask name '{mask_name}' must be a string")
                if not isinstance(file_path, str):
                    raise ValueError(f"File path for mask '{mask_name}' must be a string")
        
        self.init_parameters(rast, mask)

    def config_yaml(self):
        """
        Initialize the configuration from a YAML file.
        """
        param_file_dicts = self.get_nested('processes', self.process, 'parameters', default={})
        mask_file_dicts = self.get_nested('processes', self.process, 'masks', default={})
        
        self.init_parameters(param_file_dicts, mask_file_dicts)

    def init_parameters(self, param_file_dicts, mask_file_dicts):
        """
        Initialize the parameters based on the process configuration.
        
        Args:
            param_file_dicts: Dictionary of parameter configurations
            mask_file_dicts: Dictionary of mask configurations
        """
        self.parameter_manager.init_parameters_from_config(param_file_dicts, mask_file_dicts)

    def load_config(self):
        """Load the configuration from the YAML file."""
        if self._config is None:
            try:
                with open(self.yaml_file, 'r') as file:
                    self._config = yaml.safe_load(file)
                
                # Validate the loaded configuration
                self._validate_config()
                
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found: {self.yaml_file}")
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML syntax in configuration file: {e}")
            except Exception as e:
                raise ValueError(f"Error loading configuration: {e}")
        
        # Set the current process
        if self.process:
            self.set_current_process(self.process)

    def _validate_config(self):
        """Validate the structure and types of the loaded configuration."""
        if not isinstance(self._config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate processes section
        if 'processes' in self._config:
            self._validate_processes()
        
        # Validate other top-level sections as needed
        self._validate_global_settings()

    def _validate_processes(self):
        """Validate the processes section of the configuration."""
        processes = self._config.get('processes', {})
        if not isinstance(processes, dict):
            raise ValueError("'processes' must be a dictionary")
        
        for process_name, process_config in processes.items():
            if not isinstance(process_config, dict):
                raise ValueError(f"Process '{process_name}' must be a dictionary")
            
            # Validate parameters section
            if 'parameters' in process_config:
                params = process_config['parameters']
                if not isinstance(params, dict):
                    raise ValueError(f"Process '{process_name}': 'parameters' must be a dictionary")
            
            # Validate masks section
            if 'masks' in process_config:
                masks = process_config['masks']
                if not isinstance(masks, dict):
                    raise ValueError(f"Process '{process_name}': 'masks' must be a dictionary")
            
            # Validate pipeline section
            if 'pipeline' in process_config:
                pipeline = process_config['pipeline']
                if not isinstance(pipeline, list):
                    raise ValueError(f"Process '{process_name}': 'pipeline' must be a list")
                
                for i, step in enumerate(pipeline):
                    if not isinstance(step, dict):
                        raise ValueError(f"Process '{process_name}': pipeline step {i} must be a dictionary")
                    if 'task' not in step:
                        raise ValueError(f"Process '{process_name}': pipeline step {i} must have a 'task' field")
            
            # Validate output section
            if 'output' in process_config:
                output = process_config['output']
                if not isinstance(output, dict):
                    raise ValueError(f"Process '{process_name}': 'output' must be a dictionary")

    def _validate_global_settings(self):
        """Validate global configuration settings."""
        # Validate median settings if present
        if 'median' in self._config:
            median = self._config['median']
            if not isinstance(median, dict):
                raise ValueError("'median' must be a dictionary")
            
            if 'iterations' in median and not isinstance(median['iterations'], int):
                raise ValueError("'median.iterations' must be an integer")
            
            if 'size' in median and not isinstance(median['size'], int):
                raise ValueError("'median.size' must be an integer")
        
        # Validate other global settings as needed

    def get(self, key, default=None):
        """Get a top-level config value by key."""
        return self._config.get(key, default)

    @property
    def params(self):
        """Property to access parameters for backward compatibility."""
        return self.parameter_manager.params
    
    @params.setter
    def params(self, value):
        """Property setter to set parameters for backward compatibility."""
        self.parameter_manager.params = value

    @property  
    def curr_process(self):
        """Property to access current process for backward compatibility."""
        return self.process_manager.curr_process
    
    @curr_process.setter
    def curr_process(self, value):
        """Property setter to set current process for backward compatibility."""
        self.process_manager.curr_process = value
    