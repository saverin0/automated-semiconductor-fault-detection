from dotenv import load_dotenv
from pathlib import Path
import os
import sys
import logging
import pandas as pd
import json
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import google.auth
import numpy as np

# --- Load environment variables ---
load_dotenv()

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def setup_logger():
    """Set up a logger with both console and file output."""
    logger = logging.getLogger('prediction_db')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # Console handler for terminal logs
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs('logs', exist_ok=True)
    file_handler = logging.FileHandler('logs/prediction_db.log')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

def clean_column_name(col):
    return col.strip().replace(" ", "").replace("-", "_").replace("/", "").lower()

def json_type_to_bq_type(json_type):
    mapping = {
        "int": "INTEGER",
        "integer": "INTEGER",
        "float": "FLOAT",
        "double": "FLOAT",
        "string": "STRING",
        "varchar": "STRING",
        "bool": "BOOLEAN",
        "boolean": "BOOLEAN"
    }
    return mapping.get(json_type.lower(), "STRING")

def load_bq_schema_from_json(json_path, db_logger=None):
    try:
        if db_logger:
            db_logger.info(f"Loading BigQuery schema from {json_path}")
        with open(json_path, "r") as f:
            schema_json = json.load(f)
        fields = []
        cleaned_col_map = {}
        for col, dtype in schema_json["columns"].items():
            cleaned = clean_column_name(col)
            fields.append(bigquery.SchemaField(cleaned, json_type_to_bq_type(dtype)))
            cleaned_col_map[cleaned] = col
        if db_logger:
            db_logger.info(f"Loaded schema fields: {[f.name for f in fields]}")
        return fields, cleaned_col_map
    except Exception as e:
        if db_logger:
            db_logger.error(f"Failed to load schema from {json_path}: {e}", exc_info=True)
        raise

def create_dataset_if_not_exists(client, dataset_id, location, db_logger=None):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        if db_logger:
            db_logger.info(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        if db_logger:
            db_logger.info(f"Created dataset {dataset_id}.")

def delete_table_if_exists(client, dataset_id, table_id, db_logger=None):
    table_ref = client.dataset(dataset_id).table(table_id)
    try:
        client.get_table(table_ref)
        client.delete_table(table_ref)
        if db_logger:
            db_logger.info(f"Deleted existing table {table_id}.")
    except Exception:
        if db_logger:
            db_logger.info(f"Table {table_id} does not exist, no need to delete.")

def create_table(client, dataset_id, table_id, schema, db_logger=None):
    try:
        table_ref = client.dataset(dataset_id).table(table_id)
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        if db_logger:
            db_logger.info(f"Created table {table_id}.")
    except Exception as e:
        if db_logger:
            db_logger.error(f"Failed to create table {table_id}: {e}", exc_info=True)
        raise

def upload_good_csvs_to_bigquery(
    good_dir: str,
    project_id: str,
    dataset_id: str,
    table_id: str,
    schema,
    cleaned_col_map,
    location: str,
    write_disposition: str = "WRITE_APPEND",
    db_logger=None
):
    try:
        if db_logger:
            db_logger.info("Starting upload_good_csvs_to_bigquery process for prediction data.")
        
        # Use simple authentication (same as working training code)
        client = bigquery.Client(project=project_id, location=location)
        
        create_dataset_if_not_exists(client, dataset_id, location, db_logger=db_logger)
        delete_table_if_exists(client, dataset_id, table_id, db_logger=db_logger)
        create_table(client, dataset_id, table_id, schema, db_logger=db_logger)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        dfs = []
        for csv_file in Path(good_dir).glob("*.csv"):
            if db_logger:
                db_logger.info(f"Reading {csv_file}...")
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                if db_logger:
                    db_logger.error(f"Failed to read {csv_file}: {e}", exc_info=True)
                continue

            if df.empty:
                if db_logger:
                    db_logger.info(f"Skipped {csv_file.name}: empty DataFrame.")
                continue

            # Replace empty strings with NaN
            df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

            # Rename first column to 'wafer' if empty or unnamed (before cleaning)
            first_col = df.columns[0].strip().lower()
            if first_col in ["", "unnamed: 0"]:
                new_cols = list(df.columns)
                new_cols[0] = "wafer"
                df.columns = new_cols
                if db_logger:
                    db_logger.info(f"Renamed first column to 'wafer' in {csv_file.name}")

            # Clean and map columns using the same logic as training
            original_cols = list(df.columns)
            cleaned_cols = [clean_column_name(col) for col in original_cols]
            df.columns = cleaned_cols

            # Log original and cleaned columns
            if db_logger:
                db_logger.info(f"Original columns in {csv_file.name}: {original_cols}")
                db_logger.info(f"Cleaned columns in {csv_file.name}: {cleaned_cols}")

            # Define schema column order - only include columns that exist in DataFrame
            schema_col_order = [field.name for field in schema if field.name in df.columns]

            if db_logger:
                db_logger.info(f"Schema column order for {csv_file.name}: {schema_col_order}")

            # Check for empty wafer values
            wafer_col = "wafer"
            if wafer_col in df.columns:
                if df[wafer_col].isnull().any() or (df[wafer_col] == "").any():
                    if db_logger:
                        db_logger.info(f"Skipped {csv_file.name}: empty wafer values found.")
                    continue
            else:
                if db_logger:
                    db_logger.info(f"Skipped {csv_file.name}: wafer column missing.")
                continue

            # Log DataFrame columns before reordering
            if db_logger:
                db_logger.info(f"DataFrame columns before reordering for {csv_file.name}: {list(df.columns)}")

            # Reorder columns to match schema
            try:
                df = df[schema_col_order]
            except Exception as e:
                if db_logger:
                    db_logger.error(f"Failed to reorder columns for {csv_file.name}: {e}", exc_info=True)
                continue

            dfs.append(df)
            if db_logger:
                db_logger.info(f"Appended DataFrame from {csv_file.name} with shape {df.shape}")

        if dfs:
            big_df = pd.concat(dfs, ignore_index=True)
            if db_logger:
                db_logger.info(f"Uploading {len(big_df)} rows to BigQuery table {table_ref}...")
            try:
                job = client.load_table_from_dataframe(
                    big_df, table_ref, job_config=bigquery.LoadJobConfig(write_disposition=write_disposition)
                )
                job.result()
                if db_logger:
                    db_logger.info("Upload complete.")
                    db_logger.info(f"Uploaded columns: {list(big_df.columns)}")
                # After uploading, log columns to a separate file
                columns_log_path = f"logs/{table_id}_columns.log"
                with open(columns_log_path, "w") as f:
                    f.write(", ".join(big_df.columns))
                if db_logger:
                    db_logger.info(f"Column names logged to {columns_log_path}")
            except Exception as e:
                if db_logger:
                    db_logger.error(f"Failed to upload data to BigQuery: {e}", exc_info=True)
                raise
        else:
            if db_logger:
                db_logger.info("No data to upload.")
    except Exception as e:
        if db_logger:
            db_logger.error(f"Exception in upload_good_csvs_to_bigquery: {e}", exc_info=True)
        raise

def export_bigquery_table_to_csv(
    project_id: str,
    dataset_id: str,
    table_id: str,
    destination_csv: str,
    location: str = "US",
    db_logger=None
):
    try:
        if db_logger:
            db_logger.info(f"Exporting BigQuery table {project_id}.{dataset_id}.{table_id} to {destination_csv}")
        
        # Ensure the export directory exists
        os.makedirs(os.path.dirname(destination_csv), exist_ok=True)
        
        client = bigquery.Client(project=project_id, location=location)
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        query = f"SELECT * FROM `{table_ref}`"
        df = client.query(query).to_dataframe()
        df.to_csv(destination_csv, index=False)
        if db_logger:
            db_logger.info(f"Exported table {table_ref} to {destination_csv}")
            db_logger.info(f"Exported {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        if db_logger:
            db_logger.error(f"Failed to export table {table_ref} to CSV: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Setup logger
    db_logger = setup_logger()
    
    required_env_vars = ["PREDICTION_GOOD_DIR", "BQ_PROJECT", "BQ_DATASET", "BQ_TABLE_PREDICTION", "BQ_SCHEMA_JSON_PREDICTION"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        db_logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    good_dir = os.getenv("PREDICTION_GOOD_DIR", "data/prediction/good")
    project_id = os.getenv("BQ_PROJECT")
    dataset_id = os.getenv("BQ_DATASET")
    table_id = os.getenv("BQ_TABLE_PREDICTION")
    location = os.getenv("BQ_LOCATION", "US")
    schema_json_path = os.getenv("BQ_SCHEMA_JSON_PREDICTION")

    # Check if prediction good directory exists
    if not os.path.exists(good_dir):
        db_logger.error(f"Prediction good directory not found: {good_dir}")
        raise FileNotFoundError(f"Prediction good directory not found: {good_dir}")

    schema, cleaned_col_map = load_bq_schema_from_json(schema_json_path, db_logger=db_logger)

    upload_good_csvs_to_bigquery(
        good_dir, project_id, dataset_id, table_id, schema, cleaned_col_map, location, db_logger=db_logger
    )

    # Export to CSV in the src/exported_data_from_db folder
    export_bigquery_table_to_csv(
        project_id, dataset_id, table_id, "src/exported_data_from_db/prediction_exported_data.csv", location, db_logger=db_logger
    )