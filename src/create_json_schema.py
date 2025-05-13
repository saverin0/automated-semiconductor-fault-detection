import json
from pathlib import Path

class SchemaCreator:
    def __init__(self, output_training="schema_training.json", output_prediction="schema_prediction.json"):
        # Get the base directory path
        base_dir = Path('/workspaces/automated-semiconductor-fault-detection')

        # Create schema directory if it doesn't exist
        schema_dir = base_dir / 'schema'
        schema_dir.mkdir(exist_ok=True)

        # Set full paths for schema files
        self.output_training = schema_dir / output_training
        self.output_prediction = schema_dir / output_prediction

    def generate_schema_training(self):
        datalist_training = {
            "SampleFileName": "wafer_01012025_120000.csv",
            "LengthOfDateStampInFile": 8,
            "LengthOfTimeStampInFile": 6,
            "NumberOfColumns": 592,
            "columns": {}
        }

        # Add wafer column first
        datalist_training['columns']['wafer'] = "varchar"

        # Add sensor columns
        for i in range(1, 591):
            datalist_training['columns'][f'Sensor - {i}'] = "float"

        # Add output column
        datalist_training["columns"]["Output"] = "Integer"

        # Save the schema
        with open(self.output_training, 'w') as file:
            json.dump(datalist_training, file, indent=4)
        print(f'Training schema saved successfully at: {self.output_training}')

    def generate_schema_prediction(self):
        datalist_prediction = {
            "SampleFileName": "wafer_01012025_120000.csv",
            "LengthOfDateStampInFile": 8,
            "LengthOfTimeStampInFile": 6,
            "NumberOfColumns": 591,
            "columns": {}
        }

        # Add wafer column first
        datalist_prediction['columns']['wafer'] = "varchar"

        # Add sensor columns
        for i in range(1, 591):
            datalist_prediction['columns'][f'Sensor - {i}'] = "float"

        # Save the schema
        with open(self.output_prediction, 'w') as file:
            json.dump(datalist_prediction, file, indent=4)
        print(f'Prediction schema saved successfully at: {self.output_prediction}')

def main():
    """Main function to create both training and prediction schemas"""
    try:
        # Create schema objects
        schema_creator = SchemaCreator(
            output_training="schema_training.json",
            output_prediction="schema_prediction.json"
        )

        # Generate both schemas
        schema_creator.generate_schema_training()
        schema_creator.generate_schema_prediction()

        return True
    except Exception as e:
        print(f"Error creating schemas: {str(e)}")
        return False

if __name__ == "__main__":
    main()