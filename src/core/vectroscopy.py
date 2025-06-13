from .tile_processing import ProcessingPipeline
from .config import Config

class Vectroscopy:
    def __init__(self, config):
        self.config = config
        self.pipeline = ProcessingPipeline(self.config)

    @classmethod
    def from_config(cls, config_yaml=None):
        """
        Create an instance of Vectroscopy from a configuration file.
        
        Args:
            config_yaml (str): Path to the configuration YAML file.
        
        Returns:
            Vectroscopy: An instance of the Vectroscopy class.
        """
        config = Config(config_yaml)
        return cls(config)
    
    @classmethod
    def from_array(cls, rast=None, mask=None, crs=None, transform=None, stats=None, output=None, path=None, config_yaml: str = None):
        """
        Create an instance of Vectroscopy from an array.
        
        Args:
            rast: Single raster data or a list of raster data.
            mask: A mask to apply to the raster data.
            crs: Coordinate Reference System of the raster data.
            transform: Affine transformation for the raster data.
            config_yaml (str): Path to the configuration YAML file.
        
        Returns:
            Vectroscopy: An instance of the Vectroscopy class.
        """
        config = Config(processing="default") # could be where you have multiple processing profiles. 
        config.config_array(param=rast, mask=mask, crs=crs, transform=transform)
        return cls(config)
    
    @classmethod
    def from_file(cls, rast=None, mask=None, crs=None, transform=None, stats=None, output=None, path=None, config_yaml: str = None):
        """
        Create an instance of Vectroscopy from a file.
        
        Args:
            rast: Single raster data or a list of raster data.
            mask: A mask to apply to the raster data.
            crs: Coordinate Reference System of the raster data.
            transform: Affine transformation for the raster data.
            config_yaml (str): Path to the configuration YAML file.
        
        Returns:
            Vectroscopy: An instance of the Vectroscopy class.
        """
        config = Config()
        return cls(config)
    
    def vectorize(self):
        """
        Vectorizes data. 

        Raster data must be the same shape and have the same CRS and transform.
        
        Args:
            rasts: single raster data or a list of raster data.
            mask: A mask to apply to the raster data.
            raster_list: A list of processed raster data.
            zonal_stats: The zonal statistics for the raster data.
        
        Returns:
            List: A list of vectorized geometries.
        """
        ProcessingPipeline(self.config).process_file()
