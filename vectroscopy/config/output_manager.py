"""
Output and vectorization configuration management.
"""
import os
from pyproj import CRS


class OutputManager:
    """
    Handles output paths, drivers, and vectorization settings.
    """
    
    def __init__(self, config_instance):
        self.config = config_instance
        self.output_path = None
        self.driver = None
        self.stats = None
        self.output_filename = None
    
    def get_output_path(self):
        """Get the output path for the current process."""
        if hasattr(self, 'output_path') and self.output_path:
            return self.output_path
        
        try:
            from .process_manager import ProcessManager
            process_manager = ProcessManager(self.config)
            process = process_manager.get_current_process()
            
            if process['vectorization'].get('output_dict', ''):
                return process['vectorization']['output_dict']
            return os.getcwd()
        except (ValueError, KeyError):
            # If current process is not set, return current working directory
            return os.getcwd()

    def get_driver(self):
        """Get the driver for the current process."""
        if hasattr(self, 'driver') and self.driver:
            return self.driver
            
        try:
            from .process_manager import ProcessManager
            process_manager = ProcessManager(self.config)
            process = process_manager.get_current_process()
            
            return process['vectorization'].get('driver', 'pandas')
        except (ValueError, KeyError):
            # If current process is not set, return default driver
            return 'pandas'
    
    def create_output_filename(self):
        """Get the output filename for the current process."""
        driver = self.get_driver()
        if driver == 'pandas':
            return None
            
        extension_map = {
            'GeoJSON': 'geojson',
            'ESRI Shapefile': 'shp',
            'GPKG': 'gpkg'
        }
        file_extension = extension_map.get(driver)
        if not file_extension:
            raise ValueError(f"Unknown driver: {driver}")

        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        name = process_manager.get_current_process()["name"]
        
        # Simple sanitization: replace spaces with underscores
        safe_name = name.replace(" ", "_")
        return f"{safe_name}_final.{file_extension}"
    
    def get_output_filename(self):
        """Get the output filename for the current process."""
        return self.output_filename

    def get_cs(self, crs):
        """Get the coordinate reference system for the current process."""
        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        process = process_manager.get_current_process()
        
        cs = process['vectorization'].get('cs', None)

        crs_obj = CRS.from_string(crs) if isinstance(crs, str) else crs
        if cs is None or cs == "GCS":
            if crs_obj.is_projected:
                geogcs = crs_obj.geodetic_crs
                return geogcs.to_wkt()
            else:
                return crs_obj.to_wkt()
        elif cs == "PCS":
            if crs_obj.is_geographic:
                print("CRS is geographic, converting to projected CRS.")
                return crs_obj.to_wkt()
            else:
                return crs_obj.to_wkt()
        else:
            return cs

    def get_color(self):
        """Get the color for the current process."""
        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        process = process_manager.get_current_process()
        
        return process['vectorization'].get('color', None)

    def get_stats(self):
        """Get the statistics configuration for the current process."""
        if hasattr(self, 'stats') and self.stats:
            return self.stats
            
        try:
            from .process_manager import ProcessManager
            process_manager = ProcessManager(self.config)
            process = process_manager.get_current_process()
            
            return process['vectorization'].get('stats', [])
        except (ValueError, KeyError):
            # If current process is not set, return empty list
            return []
    
    def get_base_check(self):
        """Check if the current process is set to run in base mode."""
        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        process = process_manager.get_current_process()
        
        base_config = process['vectorization'].get('base', None)
        if isinstance(base_config, dict):
            return base_config.get('show', False)

    def get_base_stats(self):
        """Get the base statistics for the current process."""
        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        process = process_manager.get_current_process()
        
        if self.get_base_check():
            base_config = process['vectorization'].get('base', None)
            if isinstance(base_config, dict):
                return base_config.get('stats', [])
        return []
    
    def get_simplification_level(self):
        """Get the simplification level for vectorization."""
        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        process = process_manager.get_current_process()
        
        simplify = process['vectorization'].get('simplify', 0)
        return simplify
    
    def get_stack(self):
        """Check if the current process is set to stack results."""
        from .process_manager import ProcessManager
        process_manager = ProcessManager(self.config)
        process = process_manager.get_current_process()
        
        return process['vectorization'].get('stack', True)
