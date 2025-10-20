"""Configuration loader utility for spam classifier."""
import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and manages configuration for the spam classifier."""
    
    def __init__(self, config_path: str = None):
        """Initialize config loader.
        
        Args:
            config_path: Path to configuration file. Defaults to configs/baseline_config.json
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "baseline_config.json"
        
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Returns:
            Dictionary containing configuration
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Convert relative paths to absolute paths
        project_root = Path(__file__).parent.parent.parent
        for key in ['raw_data', 'processed_data', 'model_dir', 'reports_dir']:
            if key in config.get('paths', {}):
                config['paths'][key] = str(project_root / config['paths'][key])
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dots, e.g., 'model.params.C')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def save_config(self, output_path: str = None) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. Defaults to original path.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)