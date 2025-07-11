import core.tile_processing as tp
import core.vectroscopy as vp
import rasterio
import numpy as np
from osgeo import gdal, ogr, osr

def main():
    
    """From a config file"""
    config_path = r"\\lasp-store\home\taja6898\Documents\Code\mineral-mapping\config\custom_config.yaml"

    gdf = vp.Vectroscopy.from_config(config_path, process="D2300").vectorize() 
    

    """From an array"""
    path = r"\\lasp-store\home\taja6898\Documents\Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"
    mc_path = r"\\lasp-store\home\taja6898\Documents\Mars_Data\MC13_demo_parameters\MC13_BAL1_EQU_IMP_D2300.IMG"
    lat_path = r"\\lasp-store\home\taja6898\Documents\Mars_Data\LatitudeBands_demo_parameters\MC_789ABCDEFGHIJKLM_BAL1_EQU_IMP_D2300_MOS_IMP.IMG"

    # raster = gdal.Open(path)
    # band = raster.GetRasterBand(1)
    # band_array = band.ReadAsArray()
    # nodata = band.GetNoDataValue()
    # band_array[band_array == nodata] = np.nan
    # gdal_crs = raster.GetProjection()
    # gdal_transform = raster.GetGeoTransform()

    # with rasterio.open(path) as src:
    #     D2300 = src.read(1, masked=True).filled(np.nan)
    #     profile = src.profile  # get metadata
    #     transform = src.transform
    #     crs = src.crs
    # # thresholds = ['70p', '80p', '90p']
    # thresholds = ["95p", "99p"]
    # name = "D2300"

    # gdf = vp.Vectroscopy.from_array(
    #     D2300, 
    #     thresholds, 
    #     crs, 
    #     transform, 
    #     name
    # ).vectorize()

    """from specific files"""
    # vp.Vectroscopy.from_files(
    #     rast={
    #         "D2300": (path, [0.005, 0.0075, 0.01, 0.0125, 0.015])
    #     },
    #     mask=None
    # ).vectorize()


if __name__ == "__main__":
    main()