"""
Example usage of vectroscopy package.

This script demonstrates how to use vectroscopy for processing raster data.
"""

import vectroscopy as vp
import rasterio
import numpy as np

def main():
    """Example usage of vectroscopy."""
    
    print("=== Vectroscopy Example ===")
    
    # Example 1: From a config file
    print("\n1. Loading from config file...")
    config_path = r"\\lasp-store\home\taja6898\Documents\Code\vectroscopy\config\custom_config.yaml"
    
    # Uncomment to test config loading:
    # gdf = vp.Vectroscopy.from_config(config_path, process="D2300").vectorize()
    
    # Example 2: From an array
    print("\n2. Processing from array...")
    
    # Data paths (update these to your actual data paths)
    paths = {
        'windows_tile': r"\\lasp-store\home\taja6898\Documents\Mars_Data\T1250_demo_parameters\T1250_cdodtot_BAL1_D2300.IMG",
        'windows_mc': r"\\lasp-store\home\taja6898\Documents\Mars_Data\MC13_demo_parameters\MC13_BAL1_EQU_IMP_D2300.IMG",
        'mac_tile': '/Users/tahnjandai/SPATIAL DATA/T1250 Demo Parameters/T1250_cdodtot_BAL1_D2300.IMG',
        'mac_mc': '/Users/tahnjandai/SPATIAL DATA/MC13_demo_parameters/MC13_BAL1_EQU_IMP_D2300.IMG'
    }
    
    # Try to load data (adjust path based on your system)
    data_path = paths['mac_tile']  # Change this based on your system
    
    try:
        with rasterio.open(data_path) as src:
            D2300 = src.read(1, masked=True).filled(np.nan)
            transform = src.transform
            crs = src.crs
            
        print(f"Loaded data shape: {D2300.shape}")
        
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Creating synthetic data for demonstration...")
        
        # Create synthetic data
        D2300 = np.random.rand(500, 500) * 0.05
        transform = None
        crs = None
        
        print(f"Synthetic data shape: {D2300.shape}")
    
    # Processing parameters
    thresholds = [0.005, 0.0125, 0.02, 0.0275]
    
    print(f"\n3. Creating Vectroscopy instance with {len(thresholds)} thresholds...")
    
    # Create vectroscopy instance
    vp_inst = vp.Vectroscopy.from_array(
        D2300, 
        thresholds, 
        crs, 
        transform, 
        "D2300"
    # ).config_output(
    #     driver="ESRI Shapefile",
    )
    
    print("4. Running vectorization...")
    
    # Vectorize the data
    gdf = vp_inst.vectorize()
    
    print(f"✓ Vectorization complete!")
    if gdf is not None:
        print(f"  - Generated {len(gdf)} vector features")
    else:
        print("  - No vector features generated")

    print("\n=== Example Complete ===")
    
    return gdf

if __name__ == "__main__":
    result = main()
