from .processing import ProcessingPipeline
from .config import Config

class Vectroscopy:
    """Main class for handling spectroscopy data processing."""
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_config(cls, config_yaml=None, process=None):
        """
        Create an instance of Vectroscopy from a configuration file.
        
        Args:
            config_yaml (str): Path to the configuration YAML file.
            process (str): The name of the process to run. Leave as None to run all.
        
        Returns:
            Vectroscopy: An instance of the Vectroscopy class.
        """
        config = Config(config_yaml, process=process)
        return cls(config)

    @classmethod
    def from_array(cls, array, thresholds=None, crs=None, transform=None, name=None, median_iterations=1, median_size=3):
        """
        Create an instance of Vectroscopy from an array.
        
        Args:
            array: Raster data to process.
            thresholds: Threshold values for the raster data.
            crs: Coordinate Reference System of the raster data.
            transform: Affine transformation for the raster data.
            name: Name for the parameter.
            median_iterations: Number of iterations for the median filter.
            median_size: Size of the median filter.

        Returns:
            Vectroscopy: An instance of the Vectroscopy class.
        """
        config = Config(process="default")  # could be where you have multiple processing profiles.
        config.add_parameter(array=array, thresholds=thresholds, crs=crs, transform=transform, name=name, median_iterations=median_iterations, median_size=median_size)
        return cls(config)

    def add_param(self, array, thresholds=None, crs=None, transform=None, name=None):
        """
        Add another parameter to the existing configuration.
        
        Args:
            array: Raster data to add
            crs: Coordinate Reference System
            transform: Affine transformation
            name: Name for the parameter
            thresholds: Threshold values for this parameter
            
        Returns:
            self: Returns self to enable method chaining
        """
        self.config.add_parameter(array=array, crs=crs, transform=transform, name=name, thresholds=thresholds)
        return self

    def add_mask(self, array=None, crs=None, transform=None, name=None, threshold=None):
        """
        Add a mask to the existing configuration.

        Args:
            array: Raster data to add
            crs: Coordinate Reference System
            transform: Affine transformation
            name: Name for the parameter
            thresholds: Threshold values for this parameter
            
        Returns:
            self: Returns self to enable method chaining
        """
        self.config.add_mask(array=array, crs=crs, transform=transform, name=name, threshold=threshold)
        return self

    def config_output(self, stats=None, show_base=None, base_stats=None, driver=None, output_path=None):
        """
        Configure the output settings for the Vectroscopy instance.

        Args:
            stats: The statistics to include in the output.
            show_base: Whether to show the base statistics.
            base_stats: The base statistics to include.
            driver: The GDAL driver to use for output.
            output_path: The path to the output file.

        Returns:
            self: Returns self to enable method chaining.
        """
        self.config.stats = stats
        self.config.show_base = show_base
        self.config.base_stats = base_stats
        self.config.driver = driver
        self.config.output_path = output_path
        return self

    @classmethod
    def from_files(cls, rast=None, mask=None, stats=None, output=None, path=None, config_yaml: str = None):
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
        config = Config(processing="default")
        config.config_files(rast=rast, mask=mask)
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
        return ProcessingPipeline(self.config).process_file()
