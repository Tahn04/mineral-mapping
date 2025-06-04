import yaml

class config:
    """
    Configuration class for the mineral mapping application.

    Attributes:
    -----------
    yaml_file : str
        Path to the YAML configuration file.
    config : dict
        Loaded configuration settings from the YAML file.
    """
    def __init__(self, yaml_file=None):
        self.yaml_file = yaml_file or r"\\lasp-store\home\taja6898\Documents\Code\mineral-mapping\config\config.yaml"
        # self.relative_path = r"Code\mineral-mapping\config\config.yaml"
        self.load_config()
        self.crs = None
        self.transform = None
    
    def load_config(self):
        """
        Load the configuration from the YAML file.
        """
        with open(self.yaml_file, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set attributes based on the loaded config
        for key, value in self.config.items():
            setattr(self, key, value)
 

