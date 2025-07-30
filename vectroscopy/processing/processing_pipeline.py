"""
Core processing pipeline orchestration.
"""
import time
from tqdm import tqdm
from . import raster_processor as rp
from . import vectorization as vec
from .. import file_handler as fh


class ProcessingPipeline:
    """
    Main orchestrator for the complete processing pipeline.
    
    Attributes:
    -----------
        config: Configuration object containing processing parameters
    """
    def __init__(self, config):
        self.config = config
        self.crs = None
        self.transform = None
        self.mask = None
        self.indicator = False
        
        # Initialize processors
        self.raster_processor = rp.RasterProcessor(config)
        self.vectorizer = vec.Vectorizer(config)
    
    def process_file(self):
        """
        Main entry point for processing a file.
        
        Returns:
            GeoDataFrame or None: Processed vector data
        """
        try:
            fh.FileHandler()
            print(fh.FileHandler().get_directory())
            process = self.config.get_current_process()
            
            for _ in tqdm(range(1), desc=f"Processing: {process['name']}"):
                param_list = self.config.get_parameters_list()
                processed_rasters = self.process_parameters(param_list)
                return self.vectorizer.vectorize(processed_rasters, param_list, self.crs, self.transform)
                
        finally:
            fh.FileHandler().cleanup()
            print("Files cleaned up.")

    def process_parameters(self, param_list):
        """
        Process the raster data based on the configuration.

        Args:
            param_list: List of Parameter objects
            
        Returns:
            List: A list of processed raster data
        """
        start_time = time.time()

        # Separate masks from parameters
        param_list, post_processing_masks = self.raster_processor.get_post_processing_masks(param_list)
        
        # Create combined mask
        full_mask = self.raster_processor.complete_mask(param_list, post_processing_masks)
        
        # Apply thresholds
        raster_list = self.raster_processor.threshold(param_list, full_mask)
        
        end_time = time.time()
        print(f"Threshold processing execution time: {end_time - start_time:.2f} seconds")
        
        # Extract spatial information from first parameter
        target_param = param_list[0]
        self.crs = target_param.crs
        self.transform = target_param.transform

        # Apply processing pipeline (filters, etc.)
        raster_list = self.raster_processor.apply_processing_pipeline(raster_list)

        # Clean rasters with masks
        start_time = time.time()
        raster_list = self.raster_processor.clean_rasters(raster_list, param_list, full_mask)
        end_time = time.time()
        print(f"Clean rasters execution time: {end_time - start_time:.2f} seconds")
        
        return raster_list
