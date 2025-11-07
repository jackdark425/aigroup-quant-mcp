"""
Unified logging module for the quantanalyzer package
"""
import logging
import os
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (defaults to INFO or LOG_LEVEL environment variable)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level
    if level is None:
        level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
    
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def setup_file_logger(
    name: str, 
    log_file: str, 
    level: Optional[int] = None
) -> logging.Logger:
    """
    Set up a logger that writes to a file
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) 
           for h in logger.handlers):
        return logger
    
    # Set level
    if level is None:
        level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
    
    logger.setLevel(level)
    
    # Create file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger