import os
import yaml
from typing import Dict, Any

def load_config(env: str = None) -> Dict[str, Any]:
    if env is None:
        env = os.getenv('ENV', 'dev')
    
    config_path = f'config/{env}.yml'
    
    if not os.path.exists(config_path):
        config_path = 'config/base.yml'
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_config_value(key_path: str, config: Dict[str, Any] = None) -> Any:
    if config is None:
        config = load_config()
    
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    
    return value
