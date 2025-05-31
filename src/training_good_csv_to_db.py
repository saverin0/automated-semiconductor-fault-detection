import pandas as pd
from google.cloud import bigquery
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

def clean_column_name(col):
    return col.strip().replace(" ", "_").replace("-", "_").replace("/", "_").lower()

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

def create_dataset_if_not_exists(client, dataset_id, location):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        print(f"Created dataset {dataset_id}.")

def delete_table_if_exists(client, dataset_id, table_id):
    table_ref = client.dataset(dataset_id).table(table_id)
    try:
        client.get_table(table_ref)
        client.delete_table(table_ref)
        print(f"Deleted existing table {table_id}.")
    except Exception:
        print(f"Table {table_id} does not exist, no need to delete.")

def create_table(client, dataset_id, table_id, schema):
    table_ref = client.dataset(dataset_id).table(table_id)
    table = bigquery.Table(table_ref, schema=schema)
    client.create_table(table)
    print(f"Created table {table_id}.")

def upload_good_csvs_to_bigquery(
    good_dir: str,
    project_id: str,
    dataset_id: str,
    table_id: str,
    schema,
    cleaned_col_map,
    location: str,
    write_disposition: str = "WRITE_APPEND"
):
    client = bigquery.Client(project=project_id, location=location)
    create_dataset_if_not_exists(client, dataset_id, location)
    delete_table_if_exists(client, dataset_id, table_id)
    create_table(client, dataset_id, table_id, schema)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    for csv_file in Path(good_dir).glob("*.csv"):
        print(f"Uploading {csv_file} to BigQuery...")
        df = pd.read_csv(csv_file)
        if df.empty:
            print(f"Skipped {csv_file.name}: empty DataFrame.")
            continue

        # Robustly rename first column to 'wafer' if needed
        first_col = df.columns[0].strip().lower()
        if first_col in ["", "unnamed: 0", "wafer"]:
            new_cols = list(df.columns)
            new_cols[0] = "wafer"
            df.columns = new_cols

        # Clean and map columns
        original_cols = list(df.columns)
        cleaned_cols = [clean_column_name(col) for col in original_cols]
        df.columns = cleaned_cols

        # Map Good/Bad to output if present
        if "good_bad" in df.columns and "output" in cleaned_col_map:
            df = df.rename(columns={"good_bad": "output"})

        # Check for empty wafer values
        wafer_col = "wafer"  # Use "wafer" as the column name
        if wafer_col in df.columns:
            if df[wafer_col].isnull().any() or (df[wafer_col] == "").any():
                print(f"Skipped {csv_file.name}: empty wafer values found.")
                continue
        else:
            print(f"Skipped {csv_file.name}: wafer column missing.")
            continue

        # Reorder columns to match schema
        schema_col_order = [field.name for field in schema if field.name in df.columns]
        df = df[schema_col_order]

        print(f"Columns in {csv_file.name}: {list(df.columns)}")

        job = client.load_table_from_dataframe(df, table_ref)
        job.result()
        print(f"Uploaded {csv_file.name} successfully.")

if __name__ == "__main__":
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