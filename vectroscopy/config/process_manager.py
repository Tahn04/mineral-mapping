"""
Process-specific configuration management.
"""


class ProcessManager:
    """
    Handles process-specific configuration operations.
    """
    
    def __init__(self, config_instance):
        self.config = config_instance
        # Initialize curr_process from the config instance if it exists
        self.curr_process = getattr(config_instance, 'process', None)
    
    def set_current_process(self, process_name):
        """Set the current process name."""
        if not isinstance(process_name, str):
            raise ValueError("Process name must be a string")
        
        processes = self.get_processes()
        if not processes:
            raise ValueError("No processes defined in configuration")
        
        if process_name not in processes:
            available = list(processes.keys())
            raise ValueError(f"Process '{process_name}' not found. Available processes: {available}")
        
        self.curr_process = process_name
        # Also update the config instance to keep them in sync
        if hasattr(self.config, 'process'):
            self.config.process = process_name
    
    def get_processes(self):
        """Return the processes dictionary from the config."""
        return self.config._config.get('processes', {})
    
    def get_current_process(self):
        """Get the current process configuration."""
        # Try to get the current process from config if not set locally
        if self.curr_process is None:
            self.curr_process = getattr(self.config, 'process', None)
        
        # If still no process is set, try to use 'default' if it exists
        if self.curr_process is None:
            processes = self.get_processes()
            if 'default' in processes:
                self.curr_process = 'default'
                print("No process specified, using 'default' process.")
            else:
                raise ValueError("Current process is not set and no 'default' process found in configuration.")
        
        processes = self.get_processes()
        if self.curr_process not in processes:
            raise ValueError(f"Process '{self.curr_process}' not found in configuration.")
        return processes[self.curr_process]
    
    def get_nested(self, *keys, default=None):
        """
        Get a nested config value by a sequence of keys.
        Example: get_nested('processes', 'my_process', 'thresholds')
        """
        value = self.config._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_median_config(self):
        """Get the median configuration from the config."""
        try:
            curr_process = self.get_current_process()
            return self.get_nested('processes', self.curr_process, 'thresholds', 'median', default={})
        except ValueError:
            return {}

    def get_masks(self):
        """Get mask names for the current process."""
        try:
            curr_process = self.get_current_process()
            return self.get_nested('processes', self.curr_process, 'thresholds', 'masks', default={})
        except ValueError:
            return {}

    def get_pipeline(self):
        """Get the pipeline steps for the current process."""
        try:
            curr_process = self.get_current_process()
            return self.get_nested('processes', self.curr_process, 'pipeline', default=[])
        except ValueError:
            return []
    
    def get_dir_path(self):
        """Get the directory path for the current process."""
        process = self.get_current_process()
        return process.get("path", "") 
    
    def get_param_names(self):
        """Get the parameter names from the current process configuration."""
        process = self.get_current_process()
        return list(process["thresholds"]["parameters"].keys())

    def get_mask_names(self):
        """Get the mask names from the current process configuration."""
        process = self.get_current_process()
        if "masks" not in process["thresholds"] or process["thresholds"]["masks"] is None:
            print("No masks found in the process configuration.")
            return []
        return list(process["thresholds"]["masks"].keys())
    
    def get_task_param(self, task, param_name):
        """Get a specific parameter for a task in the current process pipeline."""
        if param_name in task:
            return task.get(param_name)
        else:
            return None
