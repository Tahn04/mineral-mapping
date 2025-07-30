"""
Configuration management modules for vectroscopy.

This package contains specialized managers for different aspects of configuration:
- ParameterManager: Handles parameter and mask initialization
- ProcessManager: Manages process-specific settings
- OutputManager: Handles output paths, drivers, and vectorization settings
"""

from .parameter_manager import ParameterManager
from .process_manager import ProcessManager
from .output_manager import OutputManager
from .file_utilities import FileUtilities
from .config import Config

__all__ = ['ParameterManager', 'ProcessManager', 'OutputManager', 'FileUtilities', 'Config']
