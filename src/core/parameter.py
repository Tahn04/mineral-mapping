from affine import Affine
import core.config as cfg
import core.raster_ops as ro
import core.vector_ops as vo
import core.file_handler as fh
import numpy as np
from tqdm import tqdm
import rioxarray as rxr
import xarray as xr
import dask.array as da

class Parameter:
    def __init__(self, name: str, raster_path=None, array=None, crs=None, transform=None, thresholds=None):
        self.name = name
        self.raster_path = raster_path
        self.preprocessed_path = None
        self.mask = False
        self.crs = None
        self.transform = None
        self.dataset = self.init_data(raster_path, array, crs, transform)
        self.thresholds = self.config_thresholds(thresholds)
        self.operator = None
        self.pipeline = None
        self.median_config = None

    def init_data(self, raster_path=None, array=None, crs=None, transform=None):
        """Initialize the raster data from a file or an array."""
        if raster_path:
            rx_ds = rxr.open_rasterio(raster_path, masked=True, chunks="auto")
            rx_ds = Parameter.preprocess_raster(rx_ds)
            self.crs = rx_ds.spatial_ref.crs_wkt
            transform = rx_ds.spatial_ref.GeoTransform
            self.transform = self.config_transform(transform)

            return rx_ds

        elif array is not None:
            if isinstance(transform, Affine):
                self.transform = transform
            else:
                self.transform = Affine.from_gdal(*transform) if isinstance(transform, (list, tuple)) else transform
            if hasattr(crs, 'to_wkt'):
                crs = crs.to_wkt()
            self.crs = crs 

            if not isinstance(array, da.Array):
                array = da.from_array(array, chunks="auto")
            
            if array.ndim == 2:
                height, width = array.shape
                dims = ['y', 'x']
            elif array.ndim == 3: # need to add support for 3D arrays
                bands, height, width = array.shape
                dims = ['band', 'y', 'x']
            else:
                raise ValueError("Array must be 2D or 3D")

            coords = {}
            if 'band' in dims:
                coords['band'] = range(bands)
            
            # Create spatial coordinates based on transform
            x_coords = [self.transform.c + (i + 0.5) * self.transform.a for i in range(width)]
            y_coords = [self.transform.f + (i + 0.5) * self.transform.e for i in range(height)]
            coords['x'] = x_coords
            coords['y'] = y_coords
            
            # Create DataArray
            rx_ds = xr.DataArray(
                array,
                dims=dims,
                coords=coords,
                attrs={
                    'transform': self.transform,
                    'crs': self.crs
                }
            )
            
            rx_ds.rio.write_crs(self.crs, inplace=True)
            rx_ds.rio.write_transform(self.transform, inplace=True)
            return rx_ds

        else:
            raise ValueError("Either raster_path or array with crs and transfrom must be provided.")
    @staticmethod
    def preprocess_raster(rxds):
        """Preprocess the raster data by replacing no data values with NaN.
        Transform must be an Affine object or a list/tuple in GDAL format."""
        nodata = rxds.rio.nodata
        if nodata:
            rxds = rxds.where(rxds != nodata, np.nan)
        return rxds.squeeze()
    
    def config_transform(self, transform):
        """Configure the affine transformation for the raster data."""
        if isinstance(transform, Affine):
            return transform
        elif isinstance(transform, str):
            transform_vals = [float(val) for val in transform.split()]
            if len(transform_vals) == 6:
                return Affine(transform_vals[1], transform_vals[2], transform_vals[0],
                               transform_vals[4], transform_vals[5], transform_vals[3])
        elif isinstance(transform, (list, tuple)) and len(transform) == 6:
            return Affine.from_gdal(*transform)
        else:
            return None

    # def median_filter(self, size=3, iterations=1):
    #     """Apply a median filter to the raster data."""
    #     return ro.dask_nanmedian_filter(self.dataset, window_size=size, iterations=iterations)

    def threshold(self, raster=None, thresholds=None):
        """Apply thresholds to the raster data and return a list."""
        if raster is None:
            raster = self.dataset
        if thresholds is None:
            thresholds = self.thresholds
        return ro.full_threshold(raster, thresholds)
    
    def coverage_mask(self):
        """Calculate the coverage mask for the parameter (True where raster is not NaN)."""
        return ~np.isnan(self.dataset)

    def get_median_config(self):
        """Return the median filter configuration."""
        if self.median_config:
            return self.median_config
        return {"size": 0, "iterations": 0}

    def set_thresholds(self, thresholds):
        """Set the thresholds for the parameter."""
        if isinstance(thresholds, list):
            self.thresholds = thresholds
        else:
            raise ValueError("Thresholds must be a list.")

    def set_median_filtered_path(self, raster_path):
        """Set the median filtered raster path."""
        self.median_filtered_path = raster_path

    def config_thresholds(self, thresholds):
        """Configure the thresholds for the parameter."""
        return ro.get_raster_thresholds(self.dataset, thresholds)

class Mask(Parameter):
    def __init__(self, name: str, raster_path=None, array=None, crs=None, transform=None, threshold=None):
        super().__init__(name, raster_path, array, crs, transform, threshold)
        self.mask = True
        self.keep_shape = False