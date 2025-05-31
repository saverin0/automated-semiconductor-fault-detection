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
from .data_ingestion.training_good_csv_to_db import (
    upload_good_csvs_to_bigquery,
    export_bigquery_table_to_csv,
    load_bq_schema_from_json
)
from .data_preprocessing.preprocessing import train_models
import joblib
from src.best_model_finder.tuner import Model_Finder

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

def get_db_logger(log_file: str) -> logging.Logger:
    """Setup a dedicated logger for the database process."""
    logger = logging.getLogger("db_process")
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
        'TRAINING_LOG_FILE'
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
        description='Run schema creation and data validation for wafer data (training only)',
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

    return parser.parse_args()

def run_schema_creation(config: Dict[str, str]) -> bool:
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
        schema_creator.generate_schema_training(
            filename=os.getenv("TRAINING_SCHEMA_FILENAME"),
            num_columns=int(os.getenv("TRAINING_NUM_COLUMNS"))
        )
        main_logger.info("Schema creation completed successfully")
        return True
    except Exception as e:
        main_logger.error(f"Error in schema creation: {e}", exc_info=True)
        return False

def run_data_validation(config: Dict[str, str]) -> bool:
    """Run the data validation process (training only)"""
    main_logger.info("="*60)
    main_logger.info("Starting data validation process...")
    main_logger.info("="*60)

    try:
        # FIXED: Use setup_environment() instead of TRAINING_CONFIG
        validator_config = data_validation.setup_environment()
        training_validator = data_validation.FileValidator(validator_config)
        training_validator.setup_directories()
        training_validator.validate_files_parallel()
        training_validator.summary()
        main_logger.info("Data validation completed successfully")
        return True
    except Exception as e:
        main_logger.error(f"Error in data validation: {e}", exc_info=True)
        return False

def run_database_upload_and_export() -> bool:
    """Upload good CSVs to BigQuery and export the table to CSV, with DB logging."""
    db_log_file = os.getenv("DB_LOG_FILE", "logs/db_process.log")
    db_log_file = validate_env_path(db_log_file, BASE_DIR)
    ensure_path_exists(db_log_file, is_file=True)
    db_logger = get_db_logger(str(db_log_file))

    try:
        good_dir = os.getenv("GOOD_DIR")
        project_id = os.getenv("BQ_PROJECT")
        dataset_id = os.getenv("BQ_DATASET")
        table_id = os.getenv("BQ_TABLE")
        location = os.getenv("BQ_LOCATION", "US")
        schema_json_path = os.getenv("BQ_SCHEMA_JSON")

        if not all([good_dir, project_id, dataset_id, table_id, schema_json_path]):
            db_logger.error("Missing one or more required BigQuery environment variables.")
            return False

        schema, cleaned_col_map = load_bq_schema_from_json(schema_json_path)

        db_logger.info("Uploading good CSVs to BigQuery...")
        upload_good_csvs_to_bigquery(
            good_dir, project_id, dataset_id, table_id, schema, cleaned_col_map, location, db_logger=db_logger
        )

        db_logger.info("Exporting BigQuery table to CSV...")
        export_bigquery_table_to_csv(
            project_id, dataset_id, table_id, "exported_data.csv", location, db_logger=db_logger
        )
        db_logger.info("Database upload and export completed successfully.")
        return True
    except Exception as e:
        db_logger.error(f"Database upload/export failed: {e}", exc_info=True)
        return False



def main():
    """Main function to orchestrate the entire process (training only)"""
    try:
        args = parse_args()
        config = load_config()

        main_logger.info("="*60)
        main_logger.info("Starting Wafer Data Processing Pipeline (training only)")
        main_logger.info("="*60)

        # Step 1: Schema Creation (if not skipped)
        if not args.skip_schema:
            main_logger.info("\nStep 1: Creating JSON Schema")
            if not run_schema_creation(config):
                main_logger.error("Schema creation failed. Stopping process.")
                sys.exit(1)
        else:
            main_logger.info("\nStep 1: Schema creation skipped (--skip-schema flag used)")

        # Step 2: Data Validation (if not skipped)
        if not args.skip_validation:
            main_logger.info("\nStep 2: Running Data Validation")
            if not run_data_validation(config):
                main_logger.error("Data validation failed.")
                sys.exit(1)
        else:
            main_logger.info("\nStep 2: Data validation skipped (--skip-validation flag used)")

        # Step 3: Database Upload and Export
        main_logger.info("\nStep 3: Uploading to BigQuery and exporting table")
        if not run_database_upload_and_export():
            main_logger.error("Database upload/export failed.")
            sys.exit(1)

        # Step 4: Preprocessing and Model Training
        main_logger.info("\nStep 4: Preprocessing exported data and training models")
        try:
            train_models("exported_data.csv", main_logger)
            main_logger.info("Preprocessing and model training completed successfully.")
        except Exception as e:
            main_logger.error(f"Preprocessing/model training failed: {e}", exc_info=True)
            sys.exit(1)

    except KeyboardInterrupt:
        main_logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        main_logger.error(f"Unexpected error: {e}", exc_info=True)
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