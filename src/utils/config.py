import yaml
import os
from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file.
    """
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config

    except Exception as e:
        raise CustomException("Failed to load configuration", e)
    

cfg = load_config()
