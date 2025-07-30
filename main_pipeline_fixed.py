#!/usr/bin/env python3
"""
Main Pipeline for Semiconductor Fault Detection
=========================================================

This script orchestrates the complete ML pipeline from training to prediction.

Workflow:
1. Create JSON schema for training data validation
2. Validate training CSV files (regex checks, null columns)
3. Upload good training files to GCP database
4. Export training data from database
5. Preprocess training data and train models
6. Use tuner.py to find best models for each cluster
7. Validate prediction CSV files
8. Upload prediction files to database
9. Export prediction data from database
10. Run predictions using trained models
11. Save prediction results

Usage:
    python main_pipeline.py --mode training
    python main_pipeline.py --mode prediction
    python main_pipeline.py --mode full
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_main_logger():
    """Set up main pipeline logger."""
    logger = logging.getLogger('main_pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - MAIN - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    logs_dir = os.getenv('LOGS_DIR', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.getenv('MAIN_LOG_FILE', 'main_pipeline.log')
    if os.path.isabs(log_file):
        log_path = log_file
    else:
        log_path = os.path.join(logs_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger
