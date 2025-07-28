"""
Vectroscopy: A Python package for vectorized raster data by threshold.
"""

from .vectroscopy import Vectroscopy
from .tile_processing import *
from .raster_ops import *
from .vector_ops import *
from .config import *
from .file_handler import *
from .parameter import *

__version__ = "0.1.0"
__author__ = "Tahn Jandai"
__email__ = "taja6898@colorado.edu"

__all__ = [
    "Vectroscopy",
]
