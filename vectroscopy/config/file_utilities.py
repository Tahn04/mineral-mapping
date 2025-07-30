"""
File operations and utilities for configuration management.
"""
import os
import re


class FileUtilities:
    """
    Utilities for file operations in configuration management.
    """
    
    def __init__(self, config_instance):
        self.config = config_instance
    
    def get_file_paths(self, names):
        """
        Returns the file path of the parameter raster or paths for indicators.
        
        Args:
            names: List of parameter names to find files for
            
        Returns:
            Dictionary mapping parameter names to file paths
        """
        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        
        files = os.listdir(process_manager.get_dir_path())
        files_dict = {}

        for param in names:
            file_path = self._find_file(files, param, process_manager.get_dir_path())
            if file_path:
                files_dict[param] = file_path
            else:
                print(f"File for parameter {param} not found in {process_manager.get_dir_path()}")        

        return files_dict

    def _find_file(self, files, param, dir_path):
        """
        Helper function to find the file for a given parameter in the directory.
        
        Args:
            files: List of files in the directory
            param: Parameter name to search for
            dir_path: Directory path
            
        Returns:
            Full file path if found, None otherwise
        """
        pattern = re.compile(rf".*{param}.*\.IMG$")
        for f in files:
            match = pattern.match(f)
            if match:
                return os.path.join(dir_path, f)
        return None
