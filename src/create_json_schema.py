from dotenv import load_dotenv
from pathlib import Path
import os
import json
import logging
from typing import Optional
from src.utils.path_utils import validate_env_path, ensure_path_exists

# --- Load environment variables ---
load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))
SCHEMA_LOGS_DIR = os.getenv("SCHEMA_LOGS_DIR")
SCHEMA_LOG_FILE = os.getenv("SCHEMA_LOG_FILE")
if not BASE_DIR or not SCHEMA_LOGS_DIR or not SCHEMA_LOG_FILE:
    raise RuntimeError("BASE_DIR, SCHEMA_LOGS_DIR, and SCHEMA_LOG_FILE must be set in .env")

SCHEMA_LOGS_DIR = validate_env_path(SCHEMA_LOGS_DIR, BASE_DIR)
SCHEMA_LOG_FILE = validate_env_path(SCHEMA_LOG_FILE, SCHEMA_LOGS_DIR)
ensure_path_exists(SCHEMA_LOGS_DIR)
ensure_path_exists(SCHEMA_LOG_FILE, is_file=True)

logging.basicConfig(
    filename=SCHEMA_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaCreator:
    """Creates JSON schemas for training and prediction data."""

    def __init__(self, schema_dir: Path, logger: logging.Logger):
        """
        Args:
            schema_dir (Path): Directory where schema files will be saved.
            logger (logging.Logger): Logger instance for logging.
        """
        self.schema_dir = validate_env_path(str(schema_dir), BASE_DIR)
        ensure_path_exists(self.schema_dir)
        self.logger = logger

    def generate_schema(
        self,
        filename: str,
        num_columns: int,
        include_output: bool = False
    ) -> Optional[Path]:
        """
        Generate and save a schema JSON file.

        Args:
            filename (str): Name of the schema file to create.
            num_columns (int): Number of sensor columns (including wafer).
            include_output (bool): Whether to include the Output column.
        Returns:
            Path to the created schema file, or None if failed.
        """
        try:
            output_path = self.schema_dir / filename
            schema = {
                "SampleFileName": "wafer_01012025_120000.csv",
                "LengthOfDateStampInFile": 8,
                "LengthOfTimeStampInFile": 6,
                "NumberOfColumns": num_columns,
                "columns": {}
            }
            schema['columns']['wafer'] = "varchar"
            for i in range(1, num_columns if not include_output else num_columns - 1):
                schema['columns'][f'Sensor - {i}'] = "float"
            if include_output:
                schema["columns"]["Output"] = "Integer"

            with open(output_path, 'w') as file:
                json.dump(schema, file, indent=4)
            self.logger.info(f"Schema created at: {output_path}")
            self.logger.debug(f"Schema content: {json.dumps(schema)[:500]}")  # Log first 500 chars
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to create schema {filename}: {e}", exc_info=True)
            return None

TRAINING_SCHEMA_FILENAME = os.getenv("TRAINING_SCHEMA_FILENAME")
PREDICTION_SCHEMA_FILENAME = os.getenv("PREDICTION_SCHEMA_FILENAME")
TRAINING_NUM_COLUMNS = os.getenv("TRAINING_NUM_COLUMNS")
PREDICTION_NUM_COLUMNS = os.getenv("PREDICTION_NUM_COLUMNS")

if not TRAINING_SCHEMA_FILENAME or not PREDICTION_SCHEMA_FILENAME or not TRAINING_NUM_COLUMNS or not PREDICTION_NUM_COLUMNS:
    raise RuntimeError("TRAINING_SCHEMA_FILENAME, PREDICTION_SCHEMA_FILENAME, TRAINING_NUM_COLUMNS, and PREDICTION_NUM_COLUMNS must be set in .env")

TRAINING_NUM_COLUMNS = int(TRAINING_NUM_COLUMNS)
PREDICTION_NUM_COLUMNS = int(PREDICTION_NUM_COLUMNS)

def main() -> int:
    """Main function to create both training and prediction schemas."""
    try:
        schema_dir = BASE_DIR / 'schema'
        ensure_path_exists(schema_dir)
        creator = SchemaCreator(schema_dir, logger)
        training_schema = creator.generate_schema(
            filename=TRAINING_SCHEMA_FILENAME,
            num_columns=TRAINING_NUM_COLUMNS,
            include_output=True
        )
        prediction_schema = creator.generate_schema(
            filename=PREDICTION_SCHEMA_FILENAME,
            num_columns=PREDICTION_NUM_COLUMNS,
            include_output=False
        )
        if not training_schema or not prediction_schema:
            logger.error("One or more schemas failed to be created.")
            return 1
        logger.info("Schema creation process completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error creating schemas: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())