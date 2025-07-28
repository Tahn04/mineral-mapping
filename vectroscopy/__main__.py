"""
Entry point for running vectroscopy as a module.

Usage:
    python -m vectroscopy --help
    python -m vectroscopy demo
    python -m vectroscopy process --config path/to/config.yaml
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main entry point for the vectroscopy package."""
    parser = argparse.ArgumentParser(
        description="Vectroscopy: A Python package for vectorized raster data by threshold.",
        prog="python -m vectroscopy"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demonstration of vectroscopy")
    demo_parser.add_argument("--data-path", help="Path to demo data file")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process raster data")
    process_parser.add_argument("--config", required=True, help="Path to configuration file")
    process_parser.add_argument("--output", help="Output directory")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        run_demo(args.data_path)
    elif args.command == "process":
        run_process(args.config, args.output)
    elif args.command == "version":
        show_version()
    else:
        parser.print_help()

def run_demo(data_path=None):
    """Run a demonstration of vectroscopy functionality."""
    print("Running vectroscopy demonstration...")
    
    try:
        from . import Vectroscopy
        import numpy as np
        from affine import Affine
        
        if data_path and Path(data_path).exists():
            print(f"Using data from: {data_path}")
            # Load and process real data
            import rasterio
            with rasterio.open(data_path) as src:
                data = src.read(1, masked=True).filled(np.nan)
                transform = src.transform
                crs = src.crs
        else:
            print("Using synthetic demo data...")
            # Create synthetic data for demo
            data = np.random.rand(100, 100) * 0.05
            transform = Affine.identity()  # Simple identity transform
            crs = "EPSG:4326"  # WGS84 for demo
        
        # Demo processing
        thresholds = [0.01, 0.02, 0.03]
        
        vp_inst = Vectroscopy.from_array(
            data, 
            thresholds, 
            crs, 
            transform, 
            "demo_parameter"
        ).vectorize()
        
        print(f"Created Vectroscopy instance with {len(thresholds)} thresholds")
        print(f"Generated {len(vp_inst)} vector features")
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        sys.exit(1)

def run_process(config_path, output_dir=None):
    """Process raster data using configuration file."""
    print(f"Processing with config: {config_path}")
    
    try:
        from . import Vectroscopy
        
        # Load and process using config
        gdf = Vectroscopy.from_config(config_path).vectorize()
        
        if output_dir:
            output_path = Path(output_dir) / "processed_output.shp"
            gdf.to_file(output_path)
            print(f"Results saved to: {output_path}")
        else:
            print(f"Processing completed. Generated {len(gdf)} features.")
            
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)

def show_version():
    """Show version information."""
    try:
        from . import __version__
        print(f"vectroscopy version {__version__}")
    except ImportError:
        print("vectroscopy version 0.1.0")

if __name__ == "__main__":
    main()
