import yaml

from src.database import *


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)