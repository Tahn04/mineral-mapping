import core.config as cfg
import core.raster_ops as ro
import core.vector_ops as vo
import numpy as np
from tqdm import tqdm

class Parameter:
    def __init__(self, name: str, raster_path=None, array=None, crs=None, transform=None, thresholds=None):
        self.name = name
        self.raster_path = raster_path
        self.mask = False
        self.crs = None
        self.transform = None
        self.raster = self.init_raster(raster_path, array, crs, transform)
        self.thresholds = self.config_thresholds(thresholds)

    def init_raster(self, raster_path=None, array=None, crs=None, transform=None):
        """Initialize the raster data from a file or an array."""
        if raster_path:
            dataset = ro.open_raster(raster_path)
            self.crs = dataset.crs
            self.transform = dataset.transform
            return ro.open_raster_band(dataset, 1)
        elif array is not None:
            if crs is None or transform is None:
                raise ValueError("Both crs and transform must be provided when using an array.")
            self.crs = crs # if crs is not None else cfg.Config().get('default_crs')
            self.transform = transform # if transform is not None else cfg.Config().get('default_transform')
            return array
        else:
            raise ValueError("Either raster_path or array with crs and transfrom must be provided.")

    def median_filter(self, size=3, iterations=1):
        """Apply a median filter to the raster data."""
        return ro.dask_nanmedian_filter(self.raster, window_size=size, iterations=iterations)

    def threshold(self, raster=None, thresholds=None):
        """Apply thresholds to the raster data and return a list."""
        if raster is None:
            raster = self.raster
        if thresholds is None:
            thresholds = self.thresholds
        return ro.full_threshold(raster, thresholds)
    
    def coverage_mask(self):
        """Calculate the coverage mask for the parameter (True where raster is not NaN)."""
        return ~np.isnan(self.raster)

    def get_transform(self):
        """Return the affine transform of the raster dataset."""
        return self.transform

    def get_crs(self):
        """Return the coordinate reference system of the raster dataset."""
        return self.crs
    
    def get_thresholds(self):
        """Return the thresholds for the parameter."""
        return self.thresholds
    
    def set_thresholds(self, thresholds):
        """Set the thresholds for the parameter."""
        if isinstance(thresholds, list):
            self.thresholds = thresholds
        else:
            raise ValueError("Thresholds must be a list.")

    def config_thresholds(self, thresholds):
        """Configure the thresholds for the parameter."""
        return ro.get_raster_thresholds(self.raster, thresholds)