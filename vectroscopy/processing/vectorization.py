"""
Vectorization and output handling.
"""
import os
import time
import shapely
from .. import vector_ops as vo
from .. import file_handler as fh
from .processing_utils import ColorUtils


class Vectorizer:
    """
    Handles vectorization of raster data and output operations.
    """
    
    def __init__(self, config):
        self.config = config
        self.color_utils = ColorUtils()
    
    def vectorize(self, raster_list, param_list, crs, transform):
        """
        Vectorize the raster data and apply zonal statistics.
        
        Args:
            raster_list: List of processed raster data
            param_list: List of Parameter objects
            crs: Coordinate reference system
            transform: Affine transformation
            
        Returns:
            GeoDataFrame or None: Vectorized data
        """
        simplification_level = self.config.get_simplification_level() 
        driver = self.config.get_driver()
        thresholds = self._assign_thresholds(raster_list, param_list)
        stats_list = self.config.get_stats()

        start_time = time.time()
        
        # Vectorize rasters
        gdf = vo.list_vectorize(raster_list, thresholds, crs, transform, simplification_level)
        
        # Calculate zonal statistics
        gdf = vo.list_zonal_stats(gdf, param_list, transform, stats_list)
        
        end_time = time.time()
        print(f"Vectorization took {end_time - start_time:.2f} seconds")
        
        # Apply color if specified
        color = self.config.get_color()
        if color:    
            gdf = self.color_utils.assign_color(gdf, color=color)

        # Set CRS and reproject
        gdf.set_crs(crs, inplace=True)
        cs = self.config.get_cs(crs)
        projected_gdf = gdf.to_crs(cs) 

        # Apply precision
        projected_gdf.geometry = shapely.set_precision(
            projected_gdf.geometry, 
            grid_size=0.00001
        )
        
        if driver == "pandas":
            return projected_gdf
        
        self.save_gdf(projected_gdf, driver)
        return None

    def save_gdf(self, gdf, driver):
        """
        Save the GeoDataFrame to file(s).
        
        Args:
            gdf: GeoDataFrame to save
            driver: Output driver type
        """
        stack = self.config.get_stack()
        output_dict = self.config.get_output_path()
        name = self.config.get_current_process()["name"]
        
        if stack is False:
            thresholds = gdf['Threshold'].unique()
            if len(thresholds) > 1:
                new_output_dict = os.path.join(output_dict, str(name))
                os.makedirs(new_output_dict, exist_ok=True)
                for threshold in thresholds:
                    thresh_gdf = gdf[gdf['Threshold'] == threshold]
                    filename = fh.FileHandler().create_output_filename(driver, name, threshold)
                    vo.save_gdf_to_file(thresh_gdf, new_output_dict, filename, driver=driver)
                return None
                
        filename = fh.FileHandler().create_output_filename(driver, name, "stack")
        vo.save_gdf_to_file(gdf, output_dict, filename, driver=driver)
    
    def _assign_thresholds(self, raster_list, param_list):
        """
        Assign thresholds to the raster data based on the parameters.
        
        Args:
            raster_list: List of processed raster data
            param_list: List of Parameter objects
            
        Returns:
            List of thresholds
        """
        # Check if this is an indicator process (multiple parameters combined)
        indicator = hasattr(self, 'indicator') and self.indicator
        
        if indicator:
            size = len(raster_list)
            thresholds = [i + 1 for i in range(size)]
        else:
            thresholds = param_list[0].thresholds
        
        base_check = self.config.get_base_check()
        if base_check:
            thresholds.insert(0, 0)
        
        return thresholds
