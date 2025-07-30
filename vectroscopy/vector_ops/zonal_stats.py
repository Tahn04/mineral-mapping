import geopandas as gpd
import pandas as pd
import rasterio
from exactextract import exact_extract

def list_zonal_stats(polygons, param_list, transform, stats_list):
    """
    Calculate zonal statistics for a list of polygons and parameters.
    
    Parameters:
    - polygons (list): List of polygon geometries.
    - param_list (list): List of parameters for each polygon.
    - crs: Coordinate reference system.
    - transform: Affine transform for the raster.
    
    Returns:
    - list: Zonal statistics for each polygon.
    """
    results = []

    x_res = transform.a
    y_res = abs(transform.e)  # y res is negative for north-up images
    pixel_area = x_res * y_res
    # gdf = combine_polygons(polygons[1:])
    # gdf = polygons[0:2]
    results = gpd.GeoDataFrame()
    for param in param_list:
        stats_config = config_stats(stats_list, param.name)  # Get the configured stats for the parameter
        
        # raster_path = get_tiled_raster_path(param)
        temp = zonal_stats(polygons, param, pixel_area, stats_config)
        # temp = tiled_zonal_stats(gdf, raster_path, stats_config, tile_size=2048, overlap=100, temp_dir=None, cleanup=True, strategy="raster-sequential")
        if results.empty:
            results = temp
        else:
            results = results.join(temp.set_index(results.index), rsuffix=f"_{param.name}")
            if f"geometry_{param.name}" in results.columns:
                results = results.drop(columns=[f"geometry_{param.name}"])
                # results = results.drop(columns=[f"value_{param.name}"])
    return results

def zonal_stats(gdf, param, pixel_area, stats_config):
    """ Calculate zonal statistics for a raster and vector layers."""
    if len(stats_config) != 0:
        empty_gdf = gpd.GeoDataFrame()
        param_name = param.name
        rast = param.preprocessed_path
        try:
            with rasterio.open(rast) as src:
                temp = exact_extract(
                src,
                gdf,
                stats_config,
                include_geom=True,
                include_cols="Threshold",
                # strategy="raster-sequential",
                output='pandas',
                progress=True,
                max_cells_in_memory=1000000000  # Adjust as needed for large datasets
            )
            temp = percintile_rename(temp)
            gdf = pd.concat([empty_gdf, temp], ignore_index=True)

            if pixel_area and pixel_area > 0:
                gdf[f"{param_name}_SQK"] = gdf[f"{param_name}_SQK"] * pixel_area * 0.000001
            else:
                gdf = gdf.rename(columns={f"{param_name}_SQK": f"{param_name}_CNT"})
        
            float_cols = gdf.select_dtypes(include=['float']).columns
            gdf[float_cols] = gdf[float_cols].round(5) 
        except Exception as e:
            print(f"Error calculating zonal stats for {param_name}: {e}")
    return gdf

def config_stats(stats_list, param_name):
    """configure statistics for a list of stats."""
    stat_config = []
    stats_map = {
            'mean': f"{param_name}_MEN=mean",
            'median': f"{param_name}_MDN=median",
            'area': f"{param_name}_SQK=count",
            'count': f"{param_name}_CNT=count",
            'min': f"{param_name}_MIN=min",
            'max': f"{param_name}_MAX=max",
            'std': f"{param_name}_STD=stdev",
        }
    # stats_map = {
    #         'mean': "MEN=mean",
    #         'median': "MDN=median",
    #         'area': "SQK=count",
    #         'count': "CNT=count",
    #         'min': "MIN=min",
    #         'max': "MAX=max",
    #         'std': "STD=stdev",
    #     }
    for stat in stats_list:
        if isinstance(stat, str) and stat.endswith('p'):
            if len(stat) < 2 or not stat[:-1].isdigit():
                raise ValueError(f"Invalid percentile format: {stat}. Must be a number followed by 'p'.")
            p = float(stat[:-1])
            stat_config.append(f"{param_name}=quantile(q={p/100})")
        elif stat in stats_map:
            stat_config.append(stats_map[stat])
        else:
            raise ValueError(f"Statistic '{stat}' is not supported. Supported statistics are: {list(stats_map.keys())}")
    return stat_config

def percintile_rename(gdf):
    """ Rename percentile columns in a GeoDataFrame."""
    for col in gdf.columns:
        if isinstance(col, str) and col and col[-1].isdigit():
            gdf = gdf.rename(columns={col: f"{col}P"})
    return gdf