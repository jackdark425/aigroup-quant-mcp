"""
Configuration management for the quantanalyzer package
"""
import os
from typing import Any, Dict, Optional
import json


class Config:
    """
    Configuration manager supporting multiple sources:
    1. Default values
    2. Environment variables
    3. Configuration files
    """
    
    def __init__(self):
        self._config = {}
        self._load_defaults()
        self._load_from_env()
        
    def _load_defaults(self):
        """Load default configuration values"""
        self._config = {
            # Logging
            'log_level': 'INFO',
            'log_file': None,
            
            # Performance
            'chunk_size': 10000,
            'parallel_workers': 4,
            'memory_limit': None,
            
            # Alpha158 defaults
            'alpha158_kbar': True,
            'alpha158_price': True,
            'alpha158_volume': True,
            'alpha158_rolling': True,
            'alpha158_rolling_windows': [5, 10, 20, 30, 60],
            
            # Model defaults
            'default_model_type': 'lightgbm',
            
            # Data processing
            'default_processor_chain': ['ProcessInf', 'CSZFillna', 'CSZScoreNorm'],
            
            # Performance monitoring
            'enable_performance_monitoring': False,
            'performance_log_level': 'DEBUG',
            'track_memory_usage': False
        }
        
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Logging
        if 'LOG_LEVEL' in os.environ:
            self._config['log_level'] = os.environ['LOG_LEVEL']
            
        if 'LOG_FILE' in os.environ:
            self._config['log_file'] = os.environ['LOG_FILE']
            
        # Performance
        if 'CHUNK_SIZE' in os.environ:
            self._config['chunk_size'] = int(os.environ['CHUNK_SIZE'])
            
        if 'PARALLEL_WORKERS' in os.environ:
            self._config['parallel_workers'] = int(os.environ['PARALLEL_WORKERS'])
            
        # Alpha158
        if 'ALPHA158_KBAR' in os.environ:
            self._config['alpha158_kbar'] = os.environ['ALPHA158_KBAR'].lower() == 'true'
            
        if 'ALPHA158_PRICE' in os.environ:
            self._config['alpha158_price'] = os.environ['ALPHA158_PRICE'].lower() == 'true'
            
        if 'ALPHA158_VOLUME' in os.environ:
            self._config['alpha158_volume'] = os.environ['ALPHA158_VOLUME'].lower() == 'true'
            
        if 'ALPHA158_ROLLING' in os.environ:
            self._config['alpha158_rolling'] = os.environ['ALPHA158_ROLLING'].lower() == 'true'
            
        # Performance monitoring
        if 'ENABLE_PERFORMANCE_MONITORING' in os.environ:
            self._config['enable_performance_monitoring'] = os.environ['ENABLE_PERFORMANCE_MONITORING'].lower() == 'true'
            
        if 'PERFORMANCE_LOG_LEVEL' in os.environ:
            self._config['performance_log_level'] = os.environ['PERFORMANCE_LOG_LEVEL']
            
        if 'TRACK_MEMORY_USAGE' in os.environ:
            self._config['track_memory_usage'] = os.environ['TRACK_MEMORY_USAGE'].lower() == 'true'
            
    def load_from_file(self, file_path: str):
        """
        Load configuration from a JSON file
        
        Args:
            file_path: Path to the JSON configuration file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                self._config.update(file_config)
        except FileNotFoundError:
            pass  # File not found, use defaults
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {file_path}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
        
    def set(self, key: str, value: Any):
        """
        Set a configuration value
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value
        
    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values
        
        Returns:
            Dictionary of all configuration values
        """
        return self._config.copy()


# Global configuration instance
_config = Config()


def get_config() -> Config:
    """
    Get the global configuration instance
    
    Returns:
        Global configuration instance
    """
    return _config


def reload_config(config_file: Optional[str] = None):
    """
    Reload configuration from file
    
    Args:
        config_file: Path to configuration file (optional)
    """
    global _config
    _config = Config()
    if config_file:
        _config.load_from_file(config_file)