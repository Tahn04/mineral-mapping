import core.config as cfg
import core.utils as utils
import core.processing as pr
import numpy as np
from tqdm import tqdm

class Parameter:
    def __init__(self, name: str, raster_path=None, array=None, crs=None, transform=None, thresholds=None):
        self.name = name
        self.raster_path = raster_path
        self.mask = False
        self.crs = None
        self.transform = None
        self.thresholds = thresholds
        self.raster = self.init_raster(raster_path, array, crs, transform)

    def init_raster(self, raster_path=None, array=None, crs=None, transform=None):
        """Initialize the raster data from a file or an array."""
        if raster_path:
            dataset = utils.open_raster(raster_path)
            self.crs = dataset.crs
            self.transform = dataset.transform
            return utils.open_raster_band(dataset, 1)
        elif array is not None:
            if crs is None or transform is None:
                raise ValueError("Both crs and transform must be provided when using an array.")
            self.crs = crs # if crs is not None else cfg.Config().get('default_crs')
            self.transform = transform # if transform is not None else cfg.Config().get('default_transform')
            return array
        else:
            raise ValueError("Either raster_path or array with crs and transfrom must be provided.")

    def new_median_filter(self, size=3, iterations=1):
        """Apply a median filter to the raster data."""
        return pr.dask_nanmedian_filter(self.raster, window_size=size, iterations=iterations)

    def median_filter(self, size=3, iterations=1):
        """Apply a median filter to the raster data and return a new raster."""
        return pr.bottleneck_nanmedian_filter(self.raster, window_size=size, iterations=iterations)

    def threshold(self, raster=None, thresholds=None):
        """Apply thresholds to the raster data and return a list."""
        if raster is None:
            raster = self.raster
        if thresholds is None:
            thresholds = self.thresholds
        return pr.full_threshold(raster, thresholds)
    
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
        

    # def majority_filter(self, raster_list, size=3, iterations=None):
    #     """Apply a majority filter to a list of raster data."""
    #     num_majority_filter = iterations if iterations else self.defaults.get_num_majority_filter(self.name)
    #     return pr.list_majority_filter(raster_list, size=size, iterations=num_majority_filter)

    # def boundary_clean(self, raster_list, radius=1,  iterations=None):
    #     """Apply a boundary clean filter to a list of raster data."""
    #     num_boundary_clean = iterations if iterations else self.defaults.get_num_boundary_clean(self.name)
    #     return pr.list_boundary_clean(raster_list, iterations=num_boundary_clean, radius=radius)
    
    # def sieve_filter(self, raster_list, threshold=9, iterations=3):
    #     """Apply a sieve filter to a list of raster data."""
    #     return utils.list_sieve_filter(raster_list, threshold=threshold, profile=self, iterations=iterations)
    # def get_thresholds(self):
    #     """Return the thresholds for the parameter."""
    #     return self.defaults.get_thresholds(self.name)

    # def get_num_median_filter(self):
    #     """Return the number of median filter iterations for the parameter."""
    #     return self.defaults.get_num_median_filter(self.name)

    # def get_num_majority_filter(self):
    #     """Return the number of majority filter iterations for the parameter."""
    #     return self.defaults.get_num_majority_filter(self.name)

    # def get_num_boundary_clean(self):   
    #     """Return the number of boundary clean iterations for the parameter."""
    #     return self.defaults.get_num_boundary_clean(self.name)

    # def mask_list(self, raster_list):
    #         """Return a list of masked rasters based on the coverage mask."""
    #         coverage_mask = self.coverage_mask()
    #         for i in range(len(raster_list)):
    #             raster_list[i] = utils.mask(raster_list[i], coverage_mask)
            
    #         return raster_list