import yaml
from pydantic import BaseModel
import os

class Config:
    """
    Configuration handler for the mineral mapping application.
    Loads and provides access to settings from a YAML file.
    """
    def __init__(self, yaml_file=None):
        default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config/config.yaml"))
        self.yaml_file = yaml_file or default_path
        self._config = None
        self.load_config()
        self.curr_process = None
        # self.crs = None
        # self.transform = None

    def load_config(self):
        """Load the configuration from the YAML file."""
        if self._config is None:
            with open(self.yaml_file, 'r') as file:
                self._config = yaml.safe_load(file)
        # Set top-level keys as attributes for convenience
        for key, value in self._config.items():
            setattr(self, key, value)

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
        if self.median_run_check():
            return self.get_nested('processes', self.curr_process, 'thresholds', 'median', default={})
        else:
            return False
        
    def median_run_check(self):
        """Check if the median run is enabled for the current process."""
        if self.curr_process is None:
            raise ValueError("Current process is not set.")
        return self.get_nested('processes', self.curr_process, 'thresholds', 'median', 'run', default=False)

    def get_thresholds(self, param_type, param_name):
        """Get thresholds for the current process specified by the param_type (param/mask) and name."""
        if self.curr_process is None:
            raise ValueError("Current process is not set.")
        return self.get_nested('processes', self.curr_process, 'thresholds', param_type, param_name)

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

    # Setters
    def set_current_process(self, process_name):
        """Set the current process name."""
        if process_name in self.get_processes():
            self.curr_process = process_name
        else:
            raise ValueError(f"Process '{process_name}' not found in configuration.")
    