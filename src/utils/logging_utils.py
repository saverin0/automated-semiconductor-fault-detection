"""
Centralized Logging Utility for Semiconductor Fault Detection
============================================================

This module provides a centralized logging setup for all components
of the semiconductor fault detection system.
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with both console and file output.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Log file name (relative to LOGS_DIR)
        level (int): Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        logs_dir = os.getenv('LOGS_DIR', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        log_path = os.path.join(logs_dir, log_file)
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_log_file_path(log_file_env_var):
    """
    Get the full path for a log file from environment variable.
    
    Args:
        log_file_env_var (str): Environment variable name for the log file
    
    Returns:
        str: Full path to the log file
    """
    logs_dir = os.getenv('LOGS_DIR', 'logs')
    log_file = os.getenv(log_file_env_var)
    
    if log_file:
        return os.path.join(logs_dir, log_file)
    return None

def get_file_path(env_var, default_path=None):
    """
    Get a file path from environment variable with fallback.
    
    Args:
        env_var (str): Environment variable name
        default_path (str): Default path if env var not set
    
    Returns:
        str: File path
    """
    path = os.getenv(env_var, default_path)
    if path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def get_dir_path(env_var, default_dir=None):
    """
    Get a directory path from environment variable with fallback.
    
    Args:
        env_var (str): Environment variable name
        default_dir (str): Default directory if env var not set
    
    Returns:
        str: Directory path
    """
    dir_path = os.getenv(env_var, default_dir)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    return dir_path
