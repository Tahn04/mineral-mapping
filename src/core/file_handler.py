import os
import tempfile
import shutil

class FileHandler:
    """
    A singleton class to handle file operations for mineral mapping.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FileHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.temp_dir = tempfile.mkdtemp()
        self._initialized = True

    def create_temp_file(self, prefix, suffix):
        """
        Create a temporary file in the temporary directory.

        Args:
            prefix (str): Prefix for the temporary file name.
            suffix (str): Suffix for the temporary file name.

        Returns:
            str: Path to the created temporary file.
        """
        file_path = os.path.join(self.temp_dir, f"{prefix}_temp.{suffix}")
        return file_path
    
    def get_directory(self):
        """
        Get the path to the temporary directory.

        Returns:
            str: Path to the temporary directory.
        """
        return self.temp_dir

    def cleanup(self):
        """
        Clean up the temporary directory and its contents.
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.temp_dir = None
        self._initialized = False