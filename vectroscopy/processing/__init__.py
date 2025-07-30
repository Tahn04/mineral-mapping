"""
Processing module for vectroscopy package.

This module contains the split-up processing pipeline components:
- ProcessingPipeline: Main orchestrator
- RasterProcessor: Raster operations and filtering
- Vectorizer: Vectorization and output handling
- Processing utilities and helpers
"""

from .processing_pipeline import ProcessingPipeline
from .raster_processor import RasterProcessor
from .vectorization import Vectorizer
from .processing_utils import ColorUtils, MaskUtils, ProcessingMetrics

__all__ = [
    'ProcessingPipeline',
    'RasterProcessor', 
    'Vectorizer',
    'ColorUtils',
    'MaskUtils',
    'ProcessingMetrics'
]
