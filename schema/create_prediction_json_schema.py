import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_prediction_schema(
    output_path: str,
    num_columns: int = 591,
    sample_filename: str = "wafer_08012020_120000.csv"
):
    schema = {
        "SampleFileName": sample_filename,
        "LengthOfDateStampInFile": 8,
        "LengthOfTimeStampInFile": 6,
        "NumberOfColumns": num_columns,
        "columns": {}
    }
    # Add Wafer column
    schema["columns"]["Wafer"] = "varchar"
    # Add Sensor columns
    for i in range(1, num_columns):
        schema["columns"][f"Sensor - {i}"] = "float"
    # Save the schema to a JSON file
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"Prediction schema saved to {output_path}")
    return 0

if __name__ == "__main__":
    # Example usage
    output_json = "schema/schema_prediction.json"
    generate_prediction_schema(output_json, num_columns=591, sample_filename="wafer_08012020_120000.csv")