"""
Simple test script to verify vectroscopy package imports and basic functionality.

Run this script to test that the package is properly installed and working.
"""

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        import vectroscopy as vp
        print("✓ Main package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import main package: {e}")
        return False
    
    try:
        from vectroscopy import Vectroscopy
        print("✓ Vectroscopy class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Vectroscopy class: {e}")
        return False
    
    try:
        # Test that we can access the class
        assert hasattr(vp, 'Vectroscopy')
        print("✓ Vectroscopy class accessible via package")
    except (AssertionError, AttributeError) as e:
        print(f"✗ Vectroscopy class not accessible: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality with synthetic data."""
    print("\nTesting basic functionality...")
    
    try:
        import vectroscopy as vp
        import numpy as np
        from affine import Affine
        
        # Create synthetic test data
        test_data = np.random.rand(50, 50) * 0.1
        thresholds = [0.02, 0.05, 0.08]
        
        # Create a basic transform for testing
        transform = Affine.identity()  # Simple identity transform
        crs = "EPSG:4326"  # WGS84 for testing
        
        print(f"✓ Created test data: {test_data.shape}")
        print(f"✓ Using thresholds: {thresholds}")
        print(f"✓ Created transform and CRS for testing")
        
        # Test class instantiation (avoid config-based methods for testing)
        vp_inst = vp.Vectroscopy.from_array(
            test_data, 
            thresholds, 
            crs=crs, 
            transform=transform, 
            name="test_param"
        )
        print("✓ Vectroscopy instance created successfully")
        
        # Test basic method availability
        assert hasattr(vp_inst, 'vectorize')
        print("✓ vectorize method available")
        
        # Test that the instance has expected attributes
        assert hasattr(vp_inst, 'config')
        print("✓ config attribute available")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test that key dependencies are available."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'rasterio', 'geopandas', 
        'scipy', 'tqdm', 'dask', 'xarray'
    ]
    
    all_available = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            all_available = False
    
    return all_available

def main():
    """Run all tests."""
    print("=" * 50)
    print("VECTROSCOPY PACKAGE TEST")
    print("=" * 50)
    
    # Run tests
    import_success = test_imports()
    dependency_success = test_dependencies() 
    functionality_success = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    print(f"Import Test:        {'PASS' if import_success else 'FAIL'}")
    print(f"Dependencies Test:  {'PASS' if dependency_success else 'FAIL'}")
    print(f"Functionality Test: {'PASS' if functionality_success else 'FAIL'}")
    
    overall_success = import_success and dependency_success and functionality_success
    print(f"\nOverall:            {'PASS - Package is ready to use!' if overall_success else 'FAIL - Issues detected'}")
    
    if overall_success:
        print("\n🎉 You can now use vectroscopy in your projects!")
        print("\nTry running:")
        print("  python -m vectroscopy demo")
        print("  python examples/basic_usage.py")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
