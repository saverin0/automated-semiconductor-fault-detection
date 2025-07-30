import os
import sys
import shutil
import pandas as pd
import logging
import json
import re
import datetime
from pathlib import Path

def setup_logger():
    """Set up logger for prediction data validation."""
    logger = logging.getLogger('prediction_validation')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    logs_dir = os.getenv('LOGS_DIR', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.getenv('PREDICTION_VALIDATION_LOG', 'prediction_validation.log')
    file_handler = logging.FileHandler(os.path.join(logs_dir, log_file))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_schema(schema_path):
    """Load the JSON schema for prediction data."""
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        return schema
    except Exception as e:
        raise RuntimeError(f"Failed to load schema from {schema_path}: {str(e)}")

def validate_filename(filename, pattern=r"^wafer_\d{8}_\d{6}\.csv$"):
    """Validate the filename matches the required pattern."""
    return bool(re.match(pattern, filename))

def is_valid_file(filepath, schema, logger):
    """
    Validate a prediction file against schema requirements.
    
    Args:
        filepath: Path to the CSV file
        schema: Schema dictionary with column requirements
        logger: Logger instance
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        filename = os.path.basename(filepath)
        
        # 1. Validate filename pattern
        if not validate_filename(filename):
            return False, f"Invalid filename format: {filename}"
            
        # 2. Check if file exists and is not empty
        if os.path.getsize(filepath) == 0:
            return False, f"File is empty"
            
        # 3. Check if file can be read as CSV
        df = pd.read_csv(filepath, index_col=False)
        if df.empty:
            return False, f"File has no data rows"

        # Robustly handle wafer column
        wafer_col_candidates = [col for col in df.columns if col.lower().strip() == "wafer" or col.lower().strip() == "unnamed: 0"]
        if wafer_col_candidates:
            df = df.rename(columns={wafer_col_candidates[0]: "Wafer"})

        # DEBUG: Log the actual columns for comparison
        logger.info(f"File {filename} has {len(df.columns)} columns")
        logger.info(f"First 10 CSV columns: {list(df.columns)[:10]}")
        logger.info(f"Schema expects {len(schema['columns'])} columns")
        logger.info(f"First 10 schema columns: {list(schema['columns'].keys())[:10]}")

        # Normalize columns for comparison - handle both hyphens and spaces
        def normalize_column_name(col_name):
            return col_name.lower().strip().replace(' ', '').replace('-', '').replace('_', '')

        df_columns_norm = [normalize_column_name(col) for col in df.columns]
        schema_columns_norm = [normalize_column_name(col) for col in schema["columns"].keys()]

        # Find missing columns (case-insensitive, ignore whitespace and separators)
        missing_cols = [col for col in schema_columns_norm if col not in df_columns_norm]

        # DEBUG: Show exact differences
        if len(df_columns_norm) != len(schema_columns_norm):
            logger.warning(f"Column count mismatch: CSV has {len(df_columns_norm)}, schema expects {len(schema_columns_norm)}")
            
        # Show first few normalized columns for comparison
        logger.info(f"First 5 normalized CSV columns: {df_columns_norm[:5]}")
        logger.info(f"First 5 normalized schema columns: {schema_columns_norm[:5]}")

        # 7. Report missing columns
        if missing_cols:
            missing_percent = (len(missing_cols) / len(schema_columns_norm)) * 100
            logger.warning(
                f"File {filename} missing {len(missing_cols)}/{len(schema_columns_norm)} "
                f"required columns ({missing_percent:.1f}%)"
            )
            logger.warning(f"First 10 missing columns: {missing_cols[:10]}")
            if len(missing_cols) < len(schema_columns_norm) * 0.5:
                return True, f"Missing {len(missing_cols)} columns, but below threshold"
            else:
                return False, f"Missing {len(missing_cols)}/{len(schema_columns_norm)} required columns"

        # 8. Check for completely null columns
        null_columns = []
        for col in df.columns:
            if df[col].isnull().all():
                null_columns.append(col)
        
        if len(null_columns) > 0:
            logger.warning(f"File {filename} has {len(null_columns)} completely null columns")
            
            # Option: Reject if too many null columns
            if len(null_columns) >= len(df.columns) - 1:  # All but one column are null
                return False, f"All data columns contain only null values"
        
        return True, "Valid file"
        
    except Exception as e:
        logger.error(f"Error validating {filepath}: {str(e)}")
        return False, f"Processing error: {str(e)}"

def create_error_file(filepath, error_message, bad_dir):
    """Create a file with error details."""
    filename = os.path.basename(filepath)
    error_file = os.path.join(bad_dir, f"{filename}.error")
    
    with open(error_file, 'w') as f:
        f.write(f"Error: {error_message}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Mode: prediction\n")

def main():
    logger = setup_logger()
    
    # Directories from environment variables
    input_dir = os.getenv('PREDICTION_INPUT_DIR', 'data/prediction/input')
    good_dir = os.getenv('PREDICTION_GOOD_DIR', 'data/prediction/good')
    bad_dir = os.getenv('PREDICTION_BAD_DIR', 'data/prediction/bad')
    schema_file = os.getenv('PREDICTION_SCHEMA_FILE', 'schema/schema_prediction.json')
    
    # Create directories if they don't exist
    for directory in [input_dir, good_dir, bad_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load schema
    try:
        schema = load_schema(schema_file)
        logger.info(f"Loaded schema with {len(schema['columns'])} columns")
    except Exception as e:
        logger.error(f"Failed to load schema: {str(e)}")
        return
    
    # Start validation process
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info(f"=== PREDICTION Validation started at {timestamp} ===")
    logger.info("=" * 60)
    
    # Validate each file
    files_processed = 0
    good_files = 0
    bad_files = 0
    bad_file_reasons = []
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(input_dir, filename)
        files_processed += 1
        
        logger.info(f"Processing file: {filename}")
        is_valid, message = is_valid_file(filepath, schema, logger)
        
        if is_valid:
            # Move to good directory
            shutil.copy(filepath, os.path.join(good_dir, filename))
            logger.info(f"[GOOD] {filename}: {message}")
            good_files += 1
        else:
            # Move to bad directory
            shutil.copy(filepath, os.path.join(bad_dir, filename))
            create_error_file(filepath, message, bad_dir)
            logger.warning(f"[BAD] {filename}: {message}")
            bad_files += 1
            bad_file_reasons.append((filename, message))
    
    # Print summary
    total = good_files + bad_files
    if total > 0:
        success_rate = (good_files / total) * 100
    else:
        success_rate = 0
        
    summary_msg = (
        f"\n{'=' * 60}\n"
        f"Prediction Validation Summary:\n"
        f"Total files: {total}\n"
        f"Valid files: {good_files} ({success_rate:.1f}%)\n"
        f"Invalid files: {bad_files} ({100 - success_rate:.1f}%)\n"
        f"{'=' * 60}"
    )
    logger.info(summary_msg)
    
    # Log rejection reasons
    if bad_file_reasons:
        logger.info("Rejection reasons:")
        for filename, reason in bad_file_reasons:
            logger.info(f"  {filename}: {reason}")

if __name__ == "__main__":
    main()