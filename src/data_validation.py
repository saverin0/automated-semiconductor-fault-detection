import os
import re
import shutil
import csv
import datetime
import logging
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import mmap
from dataclasses import dataclass
from threading import Lock

# --- Load environment variables from .env file ---
from dotenv import load_dotenv
load_dotenv()

# --- Configuration dataclass ---
@dataclass
class ValidatorConfig:
    input_dir: str
    good_dir: str
    bad_dir: str
    schema_file: str
    log_file: str
    filename_pattern: str = r"^wafer_\d{8}_\d{6}\.csv$"  # Strict lowercase wafer pattern
    max_workers: int = 4
    chunk_size: int = 64 * 1024
    mode: str = "training"  # 'training' or 'prediction'

    def validate(self):
        required = ['input_dir', 'good_dir', 'bad_dir', 'schema_file', 'log_file']
        missing = [f for f in required if not getattr(self, f)]
        if missing:
            raise ValueError(f"Missing required config fields: {', '.join(missing)}")
        if self.mode not in ['training', 'prediction']:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'training' or 'prediction'")

# --- Load configurations from environment ---
try:
    TRAINING_CONFIG = ValidatorConfig(
        input_dir=os.environ['TRAINING_INPUT_DIR'],
        good_dir=os.environ['TRAINING_GOOD_DIR'],
        bad_dir=os.environ['TRAINING_BAD_DIR'],
        schema_file=os.environ['TRAINING_SCHEMA_FILE'],
        log_file=os.environ['TRAINING_LOG_FILE'],
        filename_pattern=os.getenv('FILENAME_PATTERN', r"^wafer_\d{8}_\d{6}\.csv$"),
        max_workers=int(os.getenv('MAX_WORKERS', '4')),
        mode="training"
    )
    PREDICTION_CONFIG = ValidatorConfig(
        input_dir=os.environ['PREDICTION_INPUT_DIR'],
        good_dir=os.environ['PREDICTION_GOOD_DIR'],
        bad_dir=os.environ['PREDICTION_BAD_DIR'],
        schema_file=os.environ['PREDICTION_SCHEMA_FILE'],
        log_file=os.environ['PREDICTION_LOG_FILE'],
        filename_pattern=os.getenv('FILENAME_PATTERN', r"^wafer_\d{8}_\d{6}\.csv$"),
        max_workers=int(os.getenv('MAX_WORKERS', '4')),
        mode="prediction"
    )
except KeyError as e:
    raise EnvironmentError(f"Missing environment variable: {e}")

def get_logger(log_file: str, mode: str) -> logging.Logger:
    """Get a logger with a specific file handler for each mode"""
    logger = logging.getLogger(f"validator.{mode}")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def normalize_colname(name: str) -> str:
    """Normalize column names by removing spaces, dashes and converting to lowercase."""
    return name.lower().replace(' ', '').replace('-', '')

class FileValidator:
    _schema_cache: Dict[str, List[str]] = {}

    def __init__(self, config: ValidatorConfig):
        config.validate()
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.good_dir = Path(config.good_dir)
        self.bad_dir = Path(config.bad_dir)
        self.log_file = config.log_file
        self.mode = config.mode
        self.logger = get_logger(self.log_file, self.mode)
        self.regex = re.compile(config.filename_pattern)
        self._counter_lock = Lock()
        self._good_files = 0
        self._bad_files = 0
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize schema by reading from the schema file"""
        self.expected_column_names = self._read_schema(self.config.schema_file)
        self.logger.info(f"Initialized {self.mode} schema with {len(self.expected_column_names)} columns")
        self.logger.info(f"First few columns: {', '.join(self.expected_column_names[:5])}")
        self.logger.info(f"Last column: {self.expected_column_names[-1]}")

    @staticmethod
    @lru_cache(maxsize=8)
    def _read_schema(schema_file: str) -> List[str]:
        """Read schema from file and return list of column names"""
        try:
            import orjson as json_parser
        except ImportError:
            import json as json_parser

        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json_parser.loads(f.read())

        if 'columns' not in schema or not isinstance(schema['columns'], dict):
            raise ValueError(f"Schema file {schema_file} missing 'columns' dictionary")

        columns = list(schema['columns'].keys())
        return [name.strip() for name in columns]

    def setup_directories(self) -> None:
        for path in (self.good_dir, self.bad_dir):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info("="*60)
        self.logger.info(f"=== {self.mode.upper()} Validation started at {ts} ===")
        self.logger.info("="*60)

    def validate_columns(self, header: List[str]) -> Tuple[bool, Optional[str]]:
        if not header:
            return False, "Header is empty"

        # Handle empty or unnamed first column
        if header[0] == "" or header[0].lower() == "unnamed: 0":
            header[0] = "Wafer"

        # Normalize both schema and file columns
        expected_names_norm = [normalize_colname(n) for n in self.expected_column_names]
        header_names_norm = [normalize_colname(n) for n in header]

        # For training mode
        if self.mode == 'training':
            # Check if last column is either 'output' or 'good/bad'
            if header_names_norm[-1] not in ("output", "good/bad"):
                return False, f"Last column should be 'Output' or 'Good/Bad', got '{header[-1]}'"

            # Compare all columns except the last one (output/good/bad)
            header_to_compare = header_names_norm[:-1]
            expected_to_compare = expected_names_norm[:-1]

            if len(header_to_compare) != len(expected_to_compare):
                return False, (
                    f"Column count mismatch (excluding output column):\n"
                    f"Expected {len(expected_to_compare)} columns, got {len(header_to_compare)}"
                )

            mismatches = []
            for i in range(len(header_to_compare)):
                if header_to_compare[i] != expected_to_compare[i]:
                    mismatches.append(
                        f"Column {i}: Expected '{self.expected_column_names[i]}', Found '{header[i]}'"
                    )

        # For prediction mode
        else:
            if len(header_names_norm) != len(expected_names_norm):
                return False, (
                    f"Column count mismatch:\n"
                    f"Expected {len(expected_names_norm)} columns, got {len(header_names_norm)}"
                )

            mismatches = []
            for i in range(len(expected_names_norm)):
                if header_names_norm[i] != expected_names_norm[i]:
                    mismatches.append(
                        f"Column {i}: Expected '{self.expected_column_names[i]}', Found '{header[i]}'"
                    )

        if mismatches:
            mismatch_report = "\nColumn Mismatches:\n" + "\n".join(mismatches[:10])
            if len(mismatches) > 10:
                mismatch_report += f"\n... and {len(mismatches) - 10} more mismatches"
            return False, mismatch_report

        return True, None

    def _process_file(self, entry) -> None:
        start = datetime.datetime.now()
        try:
            if not self.regex.match(entry.name):
                self._reject_file(entry, f"Invalid filename format. Expected pattern: {self.config.filename_pattern}")
                return

            with open(entry.path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                try:
                    header_line = mm.readline().decode('utf-8')
                    header = next(csv.reader([header_line]))
                    valid, err = self.validate_columns(header)

                    if valid:
                        self._accept_file(entry)
                    else:
                        self._reject_file(entry, err)
                finally:
                    mm.close()
        except Exception as e:
            self._reject_file(entry, f"Error processing file: {str(e)}")
        finally:
            elapsed = (datetime.datetime.now() - start).total_seconds()
            self.logger.debug(f"[Profiling] {entry.name}: {elapsed:.4f}s")

    def _accept_file(self, entry) -> None:
        try:
            shutil.copy2(entry.path, self.good_dir / entry.name)
            with self._counter_lock:
                self._good_files += 1
            self.logger.info(f"[Good] {entry.name}")
        except Exception as e:
            self.logger.error(f"Failed to accept {entry.name}: {e}")

    def _reject_file(self, entry, reason: str) -> None:
        try:
            shutil.copy2(entry.path, self.bad_dir / entry.name)
            with self._counter_lock:
                self._bad_files += 1
            self.logger.info(f"[Bad] {entry.name}: {reason}")

            # Create detailed error file
            error_file = self.bad_dir / f"{entry.name}.error"
            with open(error_file, 'w') as f:
                f.write(f"Error: {reason}\n")
                f.write(f"Timestamp: {datetime.datetime.now()}\n")
                f.write(f"Filename Pattern: {self.config.filename_pattern}\n")
                f.write(f"Mode: {self.mode}\n")
                f.write(f"Schema File: {self.config.schema_file}\n")

                # Add file header info if available
                try:
                    with open(entry.path, 'r') as csv_file:
                        header = next(csv.reader(csv_file))
                        f.write(f"Actual Columns: {len(header)}\n")
                        f.write("First 5 columns: " + ", ".join(header[:5]) + "\n")
                        f.write("Last 5 columns: " + ", ".join(header[-5:] if len(header) >= 5 else header) + "\n")
                except Exception as e:
                    f.write(f"Could not read file header: {str(e)}\n")
        except Exception as e:
            self.logger.error(f"Failed to reject {entry.name}: {e}")

    def validate_files_parallel(self) -> None:
        entries = [e for e in os.scandir(self.input_dir) if e.is_file() and e.name.endswith('.csv')]
        if not entries:
            self.logger.warning(f"No CSV files found in {self.input_dir}")
            return

        self.logger.info(f"Found {len(entries)} CSV files to process")
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            list(executor.map(self._process_file, entries))

    def summary(self) -> None:
        total = self._good_files + self._bad_files
        if total == 0:
            self.logger.warning(f"No {self.mode} files processed.")
            print(f"No {self.mode} files processed.")
            return

        pct = (self._good_files / total) * 100
        msg = (
            f"\n{'='*60}\n"
            f"{self.mode.capitalize()} validation complete:\n"
            f"Total files processed: {total}\n"
            f"Good files: {self._good_files} ({pct:.1f}%)\n"
            f"Bad files: {self._bad_files} ({100-pct:.1f}%)\n"
            f"Schema file used: {self.config.schema_file}\n"
            f"{'='*60}"
        )
        self.logger.info(msg)
        print(msg)

def main():
    print("Starting validation process...")

    print("\nStep 1: Running Training Validation...")
    training_validator = FileValidator(TRAINING_CONFIG)
    training_validator.setup_directories()
    training_validator.validate_files_parallel()
    training_validator.summary()

    print("\nStep 2: Running Prediction Validation...")
    prediction_validator = FileValidator(PREDICTION_CONFIG)
    prediction_validator.setup_directories()
    prediction_validator.validate_files_parallel()
    prediction_validator.summary()

    print("\nValidation process complete!")

if __name__ == "__main__":
    main()