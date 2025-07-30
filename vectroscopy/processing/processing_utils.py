"""
Processing utilities and helper functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class ColorUtils:
    """
    Utilities for color assignment and visualization.
    """
    
    def __init__(self):
        self.predefined_colors = {
            'red': [255, 0, 0], 'orange': [255, 128, 0], 'yellow': [255, 255, 0],
            'lime': [128, 255, 0], 'green': [0, 255, 0], 'sea': [0, 255, 128],
            'cyan': [0, 255, 255], 'sky': [0, 128, 255], 'blue': [0, 0, 255],
            'violet': [128, 0, 255], 'magenta': [255, 0, 255], 'pink': [255, 0, 128]
        }
    
    def assign_color(self, gdf, color="viridis"):
        """
        Assign colors to geometries in GeoDataFrame based on thresholds.

        Args:
            gdf: GeoDataFrame containing the geometries
            color: Color specification (colormap name or predefined color)

        Returns:
            GeoDataFrame with added 'hex_color' column
        """
        thresholds = gdf['Threshold'].unique()
        
        if color in self.predefined_colors:
            end_color = self.predefined_colors[color]
            thresholds_sorted = sorted(thresholds)
            color_map = self._make_color_ramp(end_color, len(thresholds_sorted))
            threshold_to_color = {val: color_map[i][0] for i, val in enumerate(thresholds_sorted)}
            gdf['hex_color'] = gdf['Threshold'].map(threshold_to_color)

        elif color in plt.colormaps():
            cmap = plt.get_cmap(color, len(thresholds))
            color_map = {val: mcolors.to_hex(cmap(i)) for i, val in enumerate(sorted(thresholds))}
            gdf['hex_color'] = gdf['Threshold'].map(color_map)
        
        else:
            raise ValueError(
                f"Color '{color}' is not recognized. "
                f"Use a valid matplotlib colormap name or predefined color: {list(self.predefined_colors.keys())}"
            )

        return gdf
    
    def _make_color_ramp(self, end_color, num_thresh):
        """
        Create a color ramp from white to the specified end color.
        
        Args:
            end_color: RGB values for the end color
            num_thresh: Number of threshold levels
            
        Returns:
            List of hex color codes
        """
        r, g, b = end_color
        rs, gs, bs = (
            np.linspace(0, r, num_thresh + 1),
            np.linspace(0, g, num_thresh + 1),
            np.linspace(0, b, num_thresh + 1)
        )
        return [[self._rgb_to_hex(int(rs[i]), int(gs[i]), int(bs[i]))] for i in range(num_thresh + 1)]

    @staticmethod
    def _rgb_to_hex(r, g, b):
        """
        Convert RGB color values (0-255) to hexadecimal color code.

        Args:
            r, g, b: Red, green, blue components (0-255)

        Returns:
            Hexadecimal color code in format '#RRGGBB'
        """
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)


class MaskUtils:
    """
    Utilities for mask operations and calculations.
    """
    
    @staticmethod
    def clip_raster(raster, mask, val=1):
        """
        Mask raster data using a boolean mask or value.
        
        Args:
            raster: Input raster data
            mask: Mask to apply
            val: Value to use for masking
            
        Returns:
            Masked raster data
        """
        import xarray as xr
        
        # Create boolean mask efficiently
        if val:
            if isinstance(mask, xr.DataArray):
                bool_mask = mask != val
            else:
                bool_mask = mask != val
        else:
            bool_mask = mask
        
        # Apply mask directly
        if isinstance(raster, xr.DataArray) and isinstance(bool_mask, xr.DataArray):
            return raster.where(~bool_mask, 0)
        else:
            # Convert to numpy for efficient operations
            if isinstance(raster, xr.DataArray):
                raster_data = raster.values
            else:
                raster_data = raster
                
            if isinstance(bool_mask, xr.DataArray):
                mask_data = bool_mask.values
            else:
                mask_data = bool_mask
            
            return raster_data * ~mask_data


class ProcessingMetrics:
    """
    Utilities for tracking and reporting processing metrics.
    """
    
    def __init__(self):
        self.timings = {}
        self.metrics = {}
    
    def start_timer(self, operation_name):
        """Start timing an operation."""
        import time
        self.timings[operation_name] = {'start': time.time()}
    
    def end_timer(self, operation_name):
        """End timing an operation and store the duration."""
        import time
        if operation_name in self.timings:
            self.timings[operation_name]['end'] = time.time()
            self.timings[operation_name]['duration'] = (
                self.timings[operation_name]['end'] - self.timings[operation_name]['start']
            )
    
    def get_duration(self, operation_name):
        """Get the duration of an operation."""
        if operation_name in self.timings and 'duration' in self.timings[operation_name]:
            return self.timings[operation_name]['duration']
        return None
    
    def print_summary(self):
        """Print a summary of all timing measurements."""
        print("\n=== Processing Performance Summary ===")
        for operation, timing in self.timings.items():
            if 'duration' in timing:
                print(f"{operation}: {timing['duration']:.2f} seconds")
        print("=" * 40)
