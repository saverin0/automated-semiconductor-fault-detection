"""
Configuration management for the semiconductor fault detection project.
Centralizes all environment variables and application settings.
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management."""
    
    # Flask Configuration
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5001'))
    
    # Model Configuration
    MODEL_SAVE_DIR = os.getenv('MODEL_SAVE_DIR', 'training_model')
    MODEL_CLUSTER_PREFIX = os.getenv('MODEL_CLUSTER_PREFIX', 'model_cluster_')
    KMEANS_CLUSTERER_FILE = os.getenv('KMEANS_CLUSTERER_FILE', 'kmeans_clusterer.joblib')
    
    # Data Directories
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    RESULTS_DIR = os.getenv('RESULTS_DIR', 'prediction_results')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    EXPORTED_DATA_DIR = os.getenv('EXPORTED_DATA_DIR', 'src/exported_data_from_db')
    
    # Training Data Configuration
    TRAINING_INPUT_DIR = os.getenv('TRAINING_INPUT_DIR', 'data/training/input')
    TRAINING_GOOD_DIR = os.getenv('TRAINING_GOOD_DIR', 'data/training/good')
    TRAINING_BAD_DIR = os.getenv('TRAINING_BAD_DIR', 'data/training/bad')
    TRAINING_SCHEMA_FILE = os.getenv('TRAINING_SCHEMA_FILE', 'schema/schema_training.json')
    TRAINING_LOG_FILE = os.getenv('TRAINING_LOG_FILE', 'training_validation.log')
    
    # Prediction Data Configuration
    PREDICTION_INPUT_DIR = os.getenv('PREDICTION_INPUT_DIR', 'data/prediction/input')
    PREDICTION_GOOD_DIR = os.getenv('PREDICTION_GOOD_DIR', 'data/prediction/good')
    PREDICTION_BAD_DIR = os.getenv('PREDICTION_BAD_DIR', 'data/prediction/bad')
    PREDICTION_SCHEMA_FILE = os.getenv('PREDICTION_SCHEMA_FILE', 'schema/schema_prediction.json')
    PREDICTION_LOG_FILE = os.getenv('PREDICTION_LOG_FILE', 'prediction_validation.log')
    
    # Google Cloud Configuration
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    BIGQUERY_SERVICE_ACCOUNT = os.getenv('BIGQUERY_SERVICE_ACCOUNT')
    BQ_PROJECT = os.getenv('BQ_PROJECT')
    BQ_DATASET = os.getenv('BQ_DATASET')
    BQ_TABLE_TRAINING = os.getenv('BQ_TABLE_TRAINING')
    BQ_TABLE_PREDICTION = os.getenv('BQ_TABLE_PREDICTION')
    
    # File Patterns
    FILENAME_PATTERN = os.getenv('FILENAME_PATTERN', r'^wafer_\d{8}_\d{6}\.csv$')
    
    # Logging Configuration
    MAIN_LOG_FILE = os.getenv('MAIN_LOG_FILE', 'main_pipeline.log')
    TRAINING_PREPROCESSING_LOG = os.getenv('TRAINING_PREPROCESSING_LOG', 'training_preprocessing.log')
    PREDICTION_TEST_LOG = os.getenv('PREDICTION_TEST_LOG', 'prediction_test.log')
    
    # Model Parameters
    EXPECTED_COLUMNS = 591  # Wafer ID + 590 features
    EXPECTED_FEATURE_COLUMNS = 590  # Features only (excluding wafer ID)
    PREDICTION_THRESHOLD = 0.2  # Threshold for Good/Bad classification
    
    @classmethod
    def validate_required_config(cls) -> Dict[str, bool]:
        """Validate that all required configuration is present."""
        validation_results = {
            'flask_secret_key': bool(cls.FLASK_SECRET_KEY),
            'model_directory': Path(cls.MODEL_SAVE_DIR).exists(),
            'upload_directory': True,  # Will be created if not exists
            'results_directory': True,  # Will be created if not exists
        }
        
        # Optional Google Cloud validation
        if cls.GOOGLE_APPLICATION_CREDENTIALS:
            validation_results['google_credentials'] = Path(cls.GOOGLE_APPLICATION_CREDENTIALS).exists()
        
        return validation_results
    
    @classmethod
    def get_missing_config(cls) -> List[str]:
        """Get list of missing required configuration."""
        validation = cls.validate_required_config()
        missing = [key for key, valid in validation.items() if not valid]
        return missing
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.UPLOAD_FOLDER,
            cls.RESULTS_DIR,
            cls.LOGS_DIR,
            cls.MODEL_SAVE_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_files(cls) -> List[str]:
        """Get list of expected model files."""
        model_files = [
            cls.KMEANS_CLUSTERER_FILE
        ]
        
        # Add cluster model files
        for i in range(3):  # Assuming 3 clusters
            for model_type in ['DecisionTree', 'RandomForest', 'GradientBoosting']:
                model_files.append(f'model_cluster_{i}_{model_type}.joblib')
        
        return model_files

# Global config instance
config = Config() 