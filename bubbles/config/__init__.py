"""Configuration module for bubbles."""

from bubbles.config.loader import load_config, get_config_path
from bubbles.config.schema import Config

__all__ = ["Config", "load_config", "get_config_path"]
