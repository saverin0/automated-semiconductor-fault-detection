import json
from pathlib import Path

def generate_prediction_schema(
    output_path: str,
    num_columns: int = 591,
    sample_filename: str = "wafer_08012020_120000.csv"
):
    schema = {
        "SampleFileName": sample_filename,
        "LengthOfDateStampInFile": 8,
        "LengthOfTimeStampInFile": 6,
        "NumberofColumns": num_columns,
        "ColName": {}
    }
    # Add Wafer column
    schema["ColName"]["Wafer"] = "varchar"
    # Add Sensor columns
    for i in range(1, num_columns):
        schema["ColName"][f"Sensor - {i}"] = "float"
    # Write to file
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=4)
    print(f"Prediction schema saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    output_json = "schema/schema_prediction.json"
    generate_prediction_schema(output_json, num_columns=591, sample_filename="wafer_08012020_120000.csv")