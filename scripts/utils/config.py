"""Configuration loader utility."""
import json
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Load and manage configuration settings."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration loader."""
        if config_path is None:
            config_paths = [
                Path('configs/precision_optimized_config.json'),
                Path('configs/default_config.json')
            ]
            for path in config_paths:
                if path.exists():
                    config_path = str(path)
                    break
            else:
                config_path = 'configs/default_config.json'
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "model": {"type": "LogisticRegression"},
            "paths": {"model_dir": "models", "reports_dir": "reports"},
            "version": "1.0.0"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
