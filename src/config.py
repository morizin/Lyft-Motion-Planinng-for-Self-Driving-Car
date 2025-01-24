import os
import yaml
from typing import Dict, Any

def load_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Load configuration from a YAML file in the `configs` folder.
    
    Args:
        config_name (str): Name of the configuration file (without extension).
                          Default is "default".
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    config_path = os.path.join("configs", f"config_{config_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

# Example usage
if __name__ == "__main__":
    config = load_config("default")
    print(config)