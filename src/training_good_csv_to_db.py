import pandas as pd
from google.cloud import bigquery
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

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

def load_bq_schema_from_json(json_path):
    with open(json_path, "r") as f:
        schema_json = json.load(f)
    fields = []
    cleaned_col_map = {}
    for col, dtype in schema_json["columns"].items():
        cleaned = clean_column_name(col)
        fields.append(bigquery.SchemaField(cleaned, json_type_to_bq_type(dtype)))
        cleaned_col_map[cleaned] = col
    return fields, cleaned_col_map

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
    table_ref = client.dataset(dataset_id).table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table)
    if db_logger:
        db_logger.info(f"Created table {table_id}.")

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
    client = bigquery.Client(project=project_id, location=location)
    create_dataset_if_not_exists(client, dataset_id, location, db_logger=db_logger)
    delete_table_if_exists(client, dataset_id, table_id, db_logger=db_logger)
    create_table(client, dataset_id, table_id, schema, db_logger=db_logger)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    dfs = []
    for csv_file in Path(good_dir).glob("*.csv"):
        if db_logger:
            db_logger.info(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        if df.empty:
            if db_logger:
                db_logger.info(f"Skipped {csv_file.name}: empty DataFrame.")
            continue

        # Rename first column to 'wafer' if empty or unnamed (before cleaning)
        first_col = df.columns[0].strip().lower()
        if first_col in ["", "unnamed: 0"]:
            new_cols = list(df.columns)
            new_cols[0] = "wafer"
            df.columns = new_cols

        # Clean and map columns
        original_cols = list(df.columns)
        cleaned_cols = [clean_column_name(col) for col in original_cols]
        df.columns = cleaned_cols

        # Rename any output-like column to 'output'
        for possible_output in ["goodbad", "good_bad", "good/bad"]:
            if possible_output in df.columns:
                df = df.rename(columns={possible_output: "output"})

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
            db_logger.info(f"DataFrame columns before reordering: {list(df.columns)}")

        # Reorder columns to match schema
        schema_col_order = [field.name for field in schema]
        df = df[schema_col_order]

        dfs.append(df)

    if dfs:
        big_df = pd.concat(dfs, ignore_index=True)
        if db_logger:
            db_logger.info(f"Uploading {len(big_df)} rows to BigQuery...")
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
    else:
        if db_logger:
            db_logger.info("No data to upload.")

def export_bigquery_table_to_csv(
    project_id: str,
    dataset_id: str,
    table_id: str,
    destination_csv: str,
    location: str = "US",
    db_logger=None
):
    client = bigquery.Client(project=project_id, location=location)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref}`"
    df = client.query(query).to_dataframe()
    df.to_csv(destination_csv, index=False)
    if db_logger:
        db_logger.info(f"Exported table {table_ref} to {destination_csv}")

if __name__ == "__main__":
    required_env_vars = ["GOOD_DIR", "BQ_PROJECT", "BQ_DATASET", "BQ_TABLE", "BQ_SCHEMA_JSON"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    good_dir = os.getenv("GOOD_DIR")
    project_id = os.getenv("BQ_PROJECT")
    dataset_id = os.getenv("BQ_DATASET")
    table_id = os.getenv("BQ_TABLE")
    location = os.getenv("BQ_LOCATION", "US")
    schema_json_path = os.getenv("BQ_SCHEMA_JSON")

    schema, cleaned_col_map = load_bq_schema_from_json(schema_json_path)

    upload_good_csvs_to_bigquery(
        good_dir, project_id, dataset_id, table_id, schema, cleaned_col_map, location
    )

    export_bigquery_table_to_csv(
        project_id, dataset_id, table_id, "exported_data.csv", location
    )