import os

import core.utils as utils
import core.preprocessing as preprocessing
import numpy as np

"""
Data loading
"""
path = r"Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG"

full_raster = utils.open_raster(path)
raster = utils.open_raster_band(full_raster, 1)
# utils.show_raster(raster.read(1, masked=True), cmap='gray', title=None)

"""
Median filter
"""
meduian_filter = utils.median_kernel_filter(raster, size=3)
# utils.show_raster(meduian_filter, cmap='gray', title=None)

# print(np.nanmin(meduian_filter), np.nanmax(meduian_filter), np.nanmean(meduian_filter), np.nanstd(meduian_filter))

"""
Thresholding
"""
thresh_list = preprocessing.full_threshold(raster, "D2300")
# utils.show_raster(thresh_raster_list[4], cmap='gray', title=None) # 0.015

"""
Majority filter
"""
maj_filt_list = preprocessing.list_majority_filter(thresh_list)
# utils.show_raster(maj_filt_list[4], cmap='gray', title=None) # 0.015

"""
Boundary clean
"""
boundary_clean_list = preprocessing.list_boundary_clean(maj_filt_list)
# utils.show_raster(boundary_clean_list[4], cmap='gray', title=None) # 0.015

"""
Vectorize
"""
vector_list = preprocessing.list_vectorize(boundary_clean_list, full_raster, "D2300")
vector_stack = utils.merge_polygons(vector_list)
# utils.show_polygons(vector_stack, title=None) # 0.015

utils.save_shapefile(vector_stack, r'Code\mineral-mapping\outputs', "vectorized_output.shp")