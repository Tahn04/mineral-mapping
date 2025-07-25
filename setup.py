from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='vectroscopy',
    version='0.1.0',
    author='Tahn Jandai',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'geopandas',
        'rasterio',
        'shapely',
        'tqdm',
        'xarray',
        'dask',
        'scipy',
        'bottleneck',
        'exactextract',
    ],
)