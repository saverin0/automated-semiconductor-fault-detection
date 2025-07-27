import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from src.utils.path_utils import validate_env_path, ensure_path_exists

# --- Load environment variables ---
load_dotenv()
BASE_DIR: Path = Path(os.getenv("BASE_DIR"))
SCHEMA_LOGS_DIR: Optional[str] = os.getenv("SCHEMA_LOGS_DIR")
SCHEMA_LOG_FILE: Optional[str] = os.getenv("SCHEMA_LOG_FILE")

if not BASE_DIR or not SCHEMA_LOGS_DIR or not SCHEMA_LOG_FILE:
    raise RuntimeError("BASE_DIR, SCHEMA_LOGS_DIR, and SCHEMA_LOG_FILE must be set in .env")

SCHEMA_LOGS_DIR = validate_env_path(SCHEMA_LOGS_DIR, BASE_DIR)
SCHEMA_LOG_FILE = validate_env_path(SCHEMA_LOG_FILE, SCHEMA_LOGS_DIR)
ensure_path_exists(SCHEMA_LOGS_DIR)
ensure_path_exists(SCHEMA_LOG_FILE, is_file=True)

def setup_logging() -> logging.Logger:
    """Setup and return configured logger."""
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(
            filename=SCHEMA_LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )
    return logger

# Setup logger
logger = setup_logging()

class SchemaCreator:
    """Creates JSON schema for training data."""

    def __init__(self, schema_dir: Path, logger: logging.Logger):
        self.schema_dir = validate_env_path(str(schema_dir), BASE_DIR)
        ensure_path_exists(self.schema_dir)
        self.logger = logger

    def generate_schema_training(
        self,
        filename: str,
        num_columns: int,
        sample_filename: str = "wafer_01012025_120000.csv"
    ) -> Optional[Path]:
        """Generate and save a schema JSON file for training data."""
        try:
            output_path = self.schema_dir / filename
            schema = self._build_schema(num_columns, sample_filename)

            with open(output_path, 'w') as file:
                json.dump(schema, file, indent=4)

            self.logger.info(f"Training schema created at: {output_path}")
            self.logger.debug(f"Schema content: {json.dumps(schema)[:500]}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to create training schema {filename}: {e}", exc_info=True)
            return None

    def _build_schema(self, num_columns: int, sample_filename: str) -> Dict[str, Any]:
        """Build the schema dictionary."""
        schema = {
            "SampleFileName": sample_filename,
            "LengthOfDateStampInFile": 8,
            "LengthOfTimeStampInFile": 6,
            "NumberOfColumns": num_columns,
            "columns": {"wafer": "varchar"}
        }

        # Add sensor columns
        for i in range(1, num_columns - 1):
            schema['columns'][f'Sensor - {i}'] = "float"

        schema["columns"]["Output"] = "Integer"
        return schema

# Load additional environment variables
TRAINING_SCHEMA_FILENAME = os.getenv("TRAINING_SCHEMA_FILENAME")
TRAINING_NUM_COLUMNS = os.getenv("TRAINING_NUM_COLUMNS")

if not TRAINING_SCHEMA_FILENAME or not TRAINING_NUM_COLUMNS:
    raise RuntimeError("TRAINING_SCHEMA_FILENAME and TRAINING_NUM_COLUMNS must be set in .env")

TRAINING_NUM_COLUMNS = int(TRAINING_NUM_COLUMNS)

def main() -> int:
    """Main function to create training schema only."""
    try:
        schema_dir = BASE_DIR / 'schema'
        ensure_path_exists(schema_dir)
        creator = SchemaCreator(schema_dir, logger)
        creator.generate_schema_training(
            filename=TRAINING_SCHEMA_FILENAME,
            num_columns=TRAINING_NUM_COLUMNS
        )
        logger.info("Training schema creation process completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error creating training schema: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())