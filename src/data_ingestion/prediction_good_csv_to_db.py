import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from google.cloud import bigquery
from google.cloud.exceptions import NotFound, Conflict
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()

def load_bq_schema_from_json(schema_json_path: str) -> Tuple[List[bigquery.SchemaField], Dict[str, str]]:
    """
    Load BigQuery schema from JSON file and return schema fields with column mapping.
    
    Args:
        schema_json_path: Path to the schema JSON file
        
    Returns:
        Tuple of (schema_fields, cleaned_column_mapping)
    """
    with open(schema_json_path, 'r') as f:
        schema_data = json.load(f)
    
    schema_fields = []
    cleaned_col_map = {}
    
    for col_name, col_type in schema_data['columns'].items():
        # Clean column names for BigQuery
        clean_name = col_name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')
        cleaned_col_map[col_name] = clean_name
        
        # Map types
        if col_type.lower() in ['float', 'double']:
            bq_type = 'FLOAT'
        elif col_type.lower() in ['int', 'integer']:
            bq_type = 'INTEGER'
        else:
            bq_type = 'STRING'
        
        schema_fields.append(bigquery.SchemaField(clean_name, bq_type))
    
    return schema_fields, cleaned_col_map

def upload_prediction_csvs_to_bigquery(
    good_dir: str,
    project_id: str,
    dataset_id: str,
    table_id: str,
    schema: List[bigquery.SchemaField],
    cleaned_col_map: Dict[str, str],
    location: str = "US",
    db_logger: Optional[logging.Logger] = None
) -> bool:
    """
    Upload prediction CSV files to BigQuery.
    
    Args:
        good_dir: Directory containing valid prediction CSV files
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        schema: BigQuery schema fields
        cleaned_col_map: Mapping of original to cleaned column names
        location: BigQuery location
        db_logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if db_logger is None:
        db_logger = logging.getLogger(__name__)
    
    try:
        client = bigquery.Client(project=project_id)
        
        # Create dataset if it doesn't exist
        dataset_ref = client.dataset(dataset_id)
        try:
            client.get_dataset(dataset_ref)
            db_logger.info(f"Dataset {dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            client.create_dataset(dataset)
            db_logger.info(f"Created dataset {dataset_id}")
        
        # Create or get table
        table_ref = dataset_ref.table(table_id)
        try:
            table = client.get_table(table_ref)
            db_logger.info(f"Table {table_id} already exists")
        except NotFound:
            table = bigquery.Table(table_ref, schema=schema)
            table = client.create_table(table)
            db_logger.info(f"Created table {table_id}")
        
        # Upload CSV files
        good_path = Path(good_dir)
        csv_files = list(good_path.glob("*.csv"))
        
        if not csv_files:
            db_logger.warning(f"No CSV files found in {good_dir}")
            return True
        
        db_logger.info(f"Found {len(csv_files)} prediction CSV files to upload")
        
        for csv_file in csv_files:
            try:
                # Read and clean the CSV
                df = pd.read_csv(csv_file)
                
                # Handle unnamed first column
                if df.columns[0] in ['', 'Unnamed: 0']:
                    df.columns = ['Wafer'] + list(df.columns[1:])
                
                # Clean column names
                df.columns = [cleaned_col_map.get(col, col.lower().replace(' ', '_').replace('-', '_')) 
                             for col in df.columns]
                
                # Convert data types
                for field in schema:
                    if field.name in df.columns:
                        if field.field_type == 'FLOAT':
                            df[field.name] = pd.to_numeric(df[field.name], errors='coerce')
                        elif field.field_type == 'INTEGER':
                            df[field.name] = pd.to_numeric(df[field.name], errors='coerce').astype('Int64')
                        else:
                            df[field.name] = df[field.name].astype(str)
                
                # Add metadata columns
                df['source_file'] = csv_file.name
                df['upload_timestamp'] = pd.Timestamp.now()
                
                # Upload to BigQuery
                job_config = bigquery.LoadJobConfig(
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                    schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION]
                )
                
                job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
                job.result()  # Wait for job to complete
                
                db_logger.info(f"Uploaded {csv_file.name} ({len(df)} rows) to BigQuery")
                
            except Exception as e:
                db_logger.error(f"Failed to upload {csv_file.name}: {e}")
                continue
        
        db_logger.info("Prediction data upload to BigQuery completed")
        return True
        
    except Exception as e:
        db_logger.error(f"BigQuery upload failed: {e}")
        return False

def export_prediction_table_to_csv(
    project_id: str,
    dataset_id: str,
    table_id: str,
    output_file: str,
    location: str = "US",
    db_logger: Optional[logging.Logger] = None
) -> bool:
    """
    Export prediction table from BigQuery to CSV.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        output_file: Output CSV file path
        location: BigQuery location
        db_logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if db_logger is None:
        db_logger = logging.getLogger(__name__)
    
    try:
        client = bigquery.Client(project=project_id)
        table_ref = client.dataset(dataset_id).table(table_id)
        
        # Check if table exists
        try:
            table = client.get_table(table_ref)
        except NotFound:
            db_logger.error(f"Table {dataset_id}.{table_id} not found")
            return False
        
        # Export table to DataFrame
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
        df = client.query(query).to_dataframe()
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        db_logger.info(f"Exported {len(df)} rows from {table_id} to {output_file}")
        
        return True
        
    except Exception as e:
        db_logger.error(f"BigQuery export failed: {e}")
        return False