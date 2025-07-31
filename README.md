# Vectroscopy

**Vectroscopy** is a fast, modular Python library for thresholding raster data, combining results, and generating simplified vector outputs. It is designed for scalable processing of large geospatial datasets — including planetary and Earth-based remote sensing data — using efficient array operations, Dask-based filtering, and GDAL/RasterIO tooling.

---

## Features

- Threshold single or multiple raster bands to detect spectral features or conditions
- Combine thresholds into detection maps
- Efficient raster-to-vector conversion
- Post-process vectors with simplification, filtering, and attribute tagging
- Compute zonal statistics using fast methods (`exactextract`)
- Scales to large rasters (10GB+) with support for Dask and chunked processing

---

## Installation

Install in editable (development) mode:

```bash
git clone https://github.com/Tahn04/vectroscopy.git
cd vectroscopy

python -m venv .venv       # Create the virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

pip install -e .