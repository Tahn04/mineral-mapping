import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape

def list_vectorize(raster_list, thresholds, crs, transform, simplify_tol):
    """
    Vectorizes a list of rasters using corresponding threshold values.

    Parameters:
    - raster_list (list of np.ndarray): List of binary rasters.
    - thresholds (list of float or int): Threshold values associated with each raster.
    - crs: Coordinate Reference System (e.g., from rasterio).
    - transform: Affine transform (e.g., from rasterio).
    - simplify_tol: Simplification tolerance in map units.

    Returns:
    - List of GeoDataFrames
    """
    gdf = gpd.GeoDataFrame()
    for raster, threshold in tqdm(zip(raster_list, thresholds), desc="Vectorizing", total=len(raster_list)):
        gdf = pd.concat([gdf, vectorize_raster(raster, transform=transform, crs=crs, threshold=threshold, simplify_tol=simplify_tol)], ignore_index=True)

    return gdf

def vectorize_raster(raster, crs=None, transform=None, threshold=None, simplify_tol=0):
    """
    Convert a binary raster (numpy array or xarray.DataArray) to a GeoDataFrame with geometries.
    """
    # Convert xarray.DataArray to numpy array if needed
    if hasattr(raster, "values"):
        arr = raster.values
    else:
        arr = raster

    arr = arr.astype("uint8")
    # Ensure transform is an Affine object
    if transform is not None and not isinstance(transform, Affine):
        transform = Affine(transform[1], transform[2], transform[0],
                         transform[4], transform[5], transform[3])
    elif transform is None:
        raise ValueError("Transform must be provided.")

    results = shapes(arr, mask=arr.astype(bool), transform=transform)
    geoms = []
    vals = []
    for geom, val in results:
        if val != 0:
            poly = shape(geom)
            if simplify_tol:
                poly = poly.simplify(simplify_tol, preserve_topology=True)
            geoms.append(poly)
            vals.append(val)

    gdf = gpd.GeoDataFrame(
        {"value": vals, "Threshold": threshold, "geometry": geoms},
        crs=crs
    )
    return gdf