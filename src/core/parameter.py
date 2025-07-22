from affine import Affine
import core.config as cfg
import core.raster_ops as ro
import core.vector_ops as vo
import core.file_handler as fh
import numpy as np
from tqdm import tqdm
from osgeo import gdal, ogr, osr
import rioxarray as rxr
import xarray as xr

class Parameter:
    def __init__(self, name: str, raster_path=None, array=None, crs=None, transform=None, thresholds=None):
        self.name = name
        self.raster_path = raster_path
        self.preprocessed_path = None
        self.mask = False
        self.dataset = None
        self.crs = None
        self.transform = None
        self.raster = self.init_raster(raster_path, array, crs, transform)
        self.thresholds = self.config_thresholds(thresholds)
        self.operator = None
        self.pipeline = None
        self.median_config = None

    def init_raster(self, raster_path=None, array=None, crs=None, transform=None):
        """Initialize the raster data from a file or an array."""
        if raster_path:
            # dataset = gdal.Open(raster_path)
            # band = dataset.GetRasterBand(1)
            # band_array = band.ReadAsArray()
            # self.crs = dataset.GetProjection()
            # self.transform = dataset.GetGeoTransform()
            # if band.GetNoDataValue() is not None:
            #     nodata = band.GetNoDataValue()
            #     band_array[band_array == nodata] = np.nan
            # self.dataset = dataset
            band_array = rxr.open_rasterio(raster_path, masked=True, chunks="auto")
            band_array = Parameter.preprocess_raster(band_array)
            self.crs = band_array.spatial_ref.crs_wkt
            self.transform = band_array.spatial_ref.GeoTransform 
            # Ensure transform is a tuple/list of length 6
            if isinstance(self.transform, str):
                # Split string and convert to float
                transform_vals = [float(val) for val in self.transform.split()]
                if len(transform_vals) != 6:
                    raise ValueError("Transform string must have 6 values.")
                transform = Affine(transform_vals[1], transform_vals[2], transform_vals[0],
                         transform_vals[4], transform_vals[5], transform_vals[3])
                self.transform = transform
            else:
                raise ValueError("Transform must be a sequence of length 6 or a string with 6 space-separated values.")
            return band_array

        elif array is not None:
            if crs is None or transform is None:
                raise ValueError("Both crs and transform must be provided when using an array.")
            if hasattr(transform, 'to_gdal'):
                transform = transform.to_gdal()
            if hasattr(crs, 'to_wkt'):
                crs = crs.to_wkt()
            self.crs = crs # if crs is not None else cfg.Config().get('default_crs')
            self.transform = transform # if transform is not None else cfg.Config().get('default_transform')
            
            path = fh.FileHandler().create_temp_file(prefix=self.name, suffix='tif')
            self.raster_path = ro.save_raster_gdal(array, crs, transform, path)
            return array

        else:
            raise ValueError("Either raster_path or array with crs and transfrom must be provided.")
    @staticmethod
    def preprocess_raster(rxds):
        """Preprocess the raster data by replacing no data values with NaN."""
        nodata = rxds.rio.nodata
        if nodata:
            rxds = rxds.where(rxds != nodata, np.nan)
        return rxds.squeeze()

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
    
    def get_raster(self):
        """Return the raster data."""
        if self.raster is None:
            raise ValueError("Raster data is not initialized.")
        return self.raster

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
        return ro.get_raster_thresholds(self.raster, thresholds)
    
    def release(self):
        """Release the raster dataset."""
        self.raster = None
        self.dataset = None
        self.mask = None

class Mask(Parameter):
    def __init__(self, name: str, raster_path=None, array=None, crs=None, transform=None, thresholds=None):
        super().__init__(name, raster_path, array, crs, transform, thresholds)
        self.mask = True
        self.keep_shape = False