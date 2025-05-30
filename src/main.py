import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
from src.utils.path_utils import validate_env_path, ensure_path_exists
from src import create_json_schema
from src import data_validation

# Add the src directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# Load environment variables from .env file
load_dotenv(project_root / '.env')

BASE_DIR = project_root

def setup_main_logger() -> logging.Logger:
    """Setup a dedicated logger for the main process."""
    LOGS_DIR = os.getenv("MAIN_LOGS_DIR")
    LOG_FILE = os.getenv("MAIN_LOG_FILE")

    if not LOGS_DIR or not LOG_FILE:
        raise RuntimeError("MAIN_LOGS_DIR and MAIN_LOG_FILE must be set in .env")

    logs_dir = validate_env_path(LOGS_DIR, BASE_DIR)
    log_file = validate_env_path(LOG_FILE, logs_dir)
    ensure_path_exists(logs_dir)
    ensure_path_exists(log_file, is_file=True)

    logger = logging.getLogger("main_process")
    logger.setLevel(logging.INFO)
    # Remove any existing handlers to avoid duplicate logs
    logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_schema_logger(log_file: str) -> logging.Logger:
    """Setup a dedicated logger for the schema creation process."""
    logger = logging.getLogger("schema_creation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_config() -> Dict[str, str]:
    """Load configuration from environment variables"""
    required_vars = [
        'TRAINING_INPUT_DIR',
        'TRAINING_GOOD_DIR',
        'TRAINING_BAD_DIR',
        'TRAINING_SCHEMA_FILE',
        'TRAINING_LOG_FILE',
        'PREDICTION_INPUT_DIR',
        'PREDICTION_GOOD_DIR',
        'PREDICTION_BAD_DIR',
        'PREDICTION_SCHEMA_FILE',
        'PREDICTION_LOG_FILE'
    ]

    config = {}
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
        config[var] = value

    if missing_vars:
        msg = (
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please ensure all required variables are set in the .env file."
        )
        main_logger.error(msg, exc_info=True)
        sys.exit(1)

    return config

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run schema creation and data validation for wafer data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--skip-schema',
        action='store_true',
        help='Skip schema creation step'
    )

    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )

    parser.add_argument(
        '--mode',
        choices=['training', 'prediction', 'both'],
        default='both',
        help='Specify the validation mode (default: both)'
    )

    return parser.parse_args()

def run_schema_creation(config: Dict[str, str], mode: str) -> bool:
    """Run the schema creation process"""
    main_logger.info("="*60)
    main_logger.info("Starting schema creation process...")
    main_logger.info("="*60)

    try:
        schema_dir_env = os.getenv("SCHEMA_DIR", "schema")
        schema_dir = validate_env_path(schema_dir_env, BASE_DIR)
        ensure_path_exists(schema_dir)
        schema_log_file = os.getenv("SCHEMA_LOG_FILE", "logs/schema_creation.log")
        schema_log_file = validate_env_path(schema_log_file, BASE_DIR)
        ensure_path_exists(schema_log_file, is_file=True)
        schema_logger = get_schema_logger(str(schema_log_file))
        schema_creator = create_json_schema.SchemaCreator(
            schema_dir=schema_dir,
            logger=schema_logger
        )
        schema_creator.generate_schema_training()
        schema_creator.generate_schema_prediction()
        main_logger.info("Schema creation completed successfully")
        return True
    except Exception as e:
        main_logger.error(f"Error in schema creation: {e}", exc_info=True)
        return False

def run_data_validation(config: Dict[str, str], mode: str) -> bool:
    """Run the data validation process"""
    main_logger.info("="*60)
    main_logger.info("Starting data validation process...")
    main_logger.info("="*60)

    try:
        if mode in ['training', 'both']:
            training_validator = data_validation.FileValidator(data_validation.TRAINING_CONFIG)
            training_validator.setup_directories()
            training_validator.validate_files_parallel()
            training_validator.summary()

        if mode in ['prediction', 'both']:
            prediction_validator = data_validation.FileValidator(data_validation.PREDICTION_CONFIG)
            prediction_validator.setup_directories()
            prediction_validator.validate_files_parallel()
            prediction_validator.summary()

        main_logger.info("Data validation completed successfully")
        return True
    except Exception as e:
        main_logger.error(f"Error in data validation: {e}", exc_info=True)
        return False

def main():
    """Main function to orchestrate the entire process"""
    try:
        args = parse_args()
        config = load_config()

        main_logger.info("="*60)
        main_logger.info("Starting Wafer Data Processing Pipeline")
        main_logger.info(f"Mode: {args.mode}")
        main_logger.info("="*60)

        # Step 1: Schema Creation (if not skipped)
        if not args.skip_schema:
            main_logger.info("\nStep 1: Creating JSON Schema")
            if not run_schema_creation(config, args.mode):
                main_logger.error("Schema creation failed. Stopping process.")
                sys.exit(1)
        else:
            main_logger.info("\nStep 1: Schema creation skipped (--skip-schema flag used)")

        # Step 2: Data Validation (if not skipped)
        if not args.skip_validation:
            main_logger.info("\nStep 2: Running Data Validation")
            if not run_data_validation(config, args.mode):
                main_logger.error("Data validation failed.")
                sys.exit(1)
        else:
            main_logger.info("\nStep 2: Data validation skipped (--skip-validation flag used)")
    except KeyboardInterrupt:
        main_logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        main_logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        main_logger.error(f"Critical error in main process: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Setup main process logger
    main_logger = setup_main_logger()

    try:
        main()
    except KeyboardInterrupt:
        main_logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        main_logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)