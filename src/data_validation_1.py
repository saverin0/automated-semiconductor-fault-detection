from dotenv import load_dotenv
from pathlib import Path
import os
import logging
import re
import csv
import shutil
import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Tuple
from src.utils.path_utils import validate_env_path, ensure_path_exists
from dataclasses import dataclass
from functools import lru_cache

# --- Fallback logging setup for early errors ---
load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))
FALLBACK_LOGS_DIR = os.getenv("FALLBACK_LOGS_DIR")
FALLBACK_LOG_FILE = os.getenv("FALLBACK_LOG_FILE")
if not BASE_DIR or not FALLBACK_LOGS_DIR or not FALLBACK_LOG_FILE:
    raise RuntimeError("BASE_DIR, FALLBACK_LOGS_DIR, and FALLBACK_LOG_FILE must be set in .env")

FALLBACK_LOGS_DIR = validate_env_path(FALLBACK_LOGS_DIR, BASE_DIR)
FALLBACK_LOG_FILE = validate_env_path(FALLBACK_LOG_FILE, FALLBACK_LOGS_DIR)
ensure_path_exists(FALLBACK_LOGS_DIR)
ensure_path_exists(FALLBACK_LOG_FILE, is_file=True)

logging.basicConfig(
    filename=FALLBACK_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration dataclass ---
@dataclass
class ValidatorConfig:
    input_dir: str
    good_dir: str
    bad_dir: str
    schema_file: str
    log_file: str
    filename_pattern: str = r"^wafer_\d{8}_\d{6}\.csv$"
    max_workers: int = 4
    chunk_size: int = 64 * 1024
    mode: str = "training"

    def validate(self):
        required = ['input_dir', 'good_dir', 'bad_dir', 'schema_file', 'log_file']
        missing = [f for f in required if not getattr(self, f)]
        if missing:
            raise ValueError(f"Missing required config fields: {', '.join(missing)}")
        if self.mode not in ['training', 'prediction']:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'training' or 'prediction'")

# --- Load configurations from environment with path validation and auto-creation ---
try:
    BASE_DIR = Path(os.getenv("BASE_DIR", "/workspaces/automated-semiconductor-fault-detection"))
    TRAINING_INPUT_DIR = validate_env_path(os.environ['TRAINING_INPUT_DIR'], BASE_DIR)
    TRAINING_GOOD_DIR = validate_env_path(os.environ['TRAINING_GOOD_DIR'], BASE_DIR)
    TRAINING_BAD_DIR = validate_env_path(os.environ['TRAINING_BAD_DIR'], BASE_DIR)
    TRAINING_SCHEMA_FILE = validate_env_path(os.environ['TRAINING_SCHEMA_FILE'], BASE_DIR)
    TRAINING_LOG_FILE = validate_env_path(os.environ['TRAINING_LOG_FILE'], BASE_DIR)

    # Ensure directories and files exist
    ensure_path_exists(TRAINING_INPUT_DIR)
    ensure_path_exists(TRAINING_GOOD_DIR)
    ensure_path_exists(TRAINING_BAD_DIR)
    ensure_path_exists(TRAINING_SCHEMA_FILE, is_file=True)
    ensure_path_exists(TRAINING_LOG_FILE, is_file=True)

    TRAINING_CONFIG = ValidatorConfig(
        input_dir=str(TRAINING_INPUT_DIR),
        good_dir=str(TRAINING_GOOD_DIR),
        bad_dir=str(TRAINING_BAD_DIR),
        schema_file=str(TRAINING_SCHEMA_FILE),
        log_file=str(TRAINING_LOG_FILE),
        filename_pattern=os.getenv('FILENAME_PATTERN', r"^wafer_\d{8}_\d{6}\.csv$"),
        max_workers=int(os.getenv('MAX_WORKERS', '4')),
        mode="training"
    )
    PREDICTION_INPUT_DIR = validate_env_path(os.environ['PREDICTION_INPUT_DIR'], BASE_DIR)
    PREDICTION_GOOD_DIR = validate_env_path(os.environ['PREDICTION_GOOD_DIR'], BASE_DIR)
    PREDICTION_BAD_DIR = validate_env_path(os.environ['PREDICTION_BAD_DIR'], BASE_DIR)
    PREDICTION_SCHEMA_FILE = validate_env_path(os.environ['PREDICTION_SCHEMA_FILE'], BASE_DIR)
    PREDICTION_LOG_FILE = validate_env_path(os.environ['PREDICTION_LOG_FILE'], BASE_DIR)

    ensure_path_exists(PREDICTION_INPUT_DIR)
    ensure_path_exists(PREDICTION_GOOD_DIR)
    ensure_path_exists(PREDICTION_BAD_DIR)
    ensure_path_exists(PREDICTION_SCHEMA_FILE, is_file=True)
    ensure_path_exists(PREDICTION_LOG_FILE, is_file=True)

    PREDICTION_CONFIG = ValidatorConfig(
        input_dir=str(PREDICTION_INPUT_DIR),
        good_dir=str(PREDICTION_GOOD_DIR),
        bad_dir=str(PREDICTION_BAD_DIR),
        schema_file=str(PREDICTION_SCHEMA_FILE),
        log_file=str(PREDICTION_LOG_FILE),
        filename_pattern=os.getenv('FILENAME_PATTERN', r"^wafer_\d{8}_\d{6}\.csv$"),
        max_workers=int(os.getenv('MAX_WORKERS', '4')),
        mode="prediction"
    )
except (KeyError, ValueError) as e:
    msg = f"Error: Invalid or missing environment variable: {e}. Please check your .env file."
    logging.error(msg, exc_info=True)
    raise RuntimeError(str(e))

def get_logger(log_file: str, mode: str) -> logging.Logger:
    """Get a logger with a specific file handler for each mode"""
    logger = logging.getLogger(f"validator.{mode}")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def normalize_colname(name: str) -> str:
    """Normalize column names by removing spaces, dashes and converting to lowercase."""
    return name.lower().replace(' ', '').replace('-', '')

def hash_columns(columns: list) -> str:
    """Hash the ordered list of column names for fast comparison."""
    import hashlib
    joined = '|'.join(columns)
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()

class FileValidator:
    _schema_cache: Dict[str, List[str]] = {}

    def __init__(self, config: ValidatorConfig):
        try:
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
            self._bad_file_reasons = []  # Track (filename, reason)
            self.logger.info(f"Validator initialized for mode: {self.mode}")
            self._init_schema()
        except Exception as e:
            self.logger.error(f"Failed to initialize validator: {e}", exc_info=True)
            raise RuntimeError(str(e))

    def _init_schema(self) -> None:
        try:
            self.expected_column_names = self._read_schema(self.config.schema_file)
            self.logger.info(f"Initialized {self.mode} schema with {len(self.expected_column_names)} columns")
            self.logger.info(f"First few columns: {', '.join(self.expected_column_names[:5])}")
            self.logger.info(f"Last column: {self.expected_column_names[-1]}")
        except Exception as e:
            self.logger.error(f"Failed to initialize schema: {e}", exc_info=True)
            raise RuntimeError(str(e))

    @staticmethod
    @lru_cache(maxsize=8)
    def _read_schema(schema_file: str) -> List[str]:
        try:
            import orjson as json_parser
        except ImportError:
            import json as json_parser

        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema = json_parser.loads(f.read())
        except Exception as e:
            logging.error(f"Failed to read schema file {schema_file}: {e}", exc_info=True)
            raise

        if 'columns' not in schema or not isinstance(schema['columns'], dict):
            logging.error(f"Schema file {schema_file} missing 'columns' dictionary", exc_info=True)
            raise ValueError(f"Schema file {schema_file} missing 'columns' dictionary")

        columns = list(schema['columns'].keys())
        return [name.strip() for name in columns]

    def setup_directories(self) -> None:
        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            raise RuntimeError(f"Input directory does not exist: {self.input_dir}")
        else:
            self.logger.info(f"Input directory exists: {self.input_dir}")

        for path in (self.good_dir, self.bad_dir):
            try:
                if path.exists():
                    shutil.rmtree(path)
                    self.logger.info(f"Removed existing directory: {path}")
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {path}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
                raise RuntimeError(str(e))

        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info("="*60)
        self.logger.info(f"=== {self.mode.upper()} Validation started at {ts} ===")
        self.logger.info("="*60)

    def validate_columns(self, header: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that the columns in the header match the expected schema columns.
        Uses hashing for fast comparison. If hashes do not match, falls back to detailed comparison (currently commented).
        """
        if not header:
            return False, "Header is empty"

        if header[0] == "" or header[0].lower() == "unnamed: 0":
            header[0] = "Wafer"

        expected_names_norm = [normalize_colname(n) for n in self.expected_column_names]
        header_names_norm = [normalize_colname(n) for n in header]

        if self.mode == 'training':
            if header_names_norm[-1] not in ("output", "good/bad"):
                self.logger.info(f"[Bad] {header}: Last column should be 'Output' or 'Good/Bad', got '{header[-1]}'")
                return False, f"Last column should be 'Output' or 'Good/Bad', got '{header[-1]}'"
            header_to_compare = header_names_norm[:-1]
            expected_to_compare = expected_names_norm[:-1]
        else:
            header_to_compare = header_names_norm
            expected_to_compare = expected_names_norm

        # --- Fast hash-based comparison ---
        # This is the fastest and most robust way for large schemas.
        if hash_columns(header_to_compare) == hash_columns(expected_to_compare):
            self.logger.info("Column validation passed using hash comparison.")
            return True, None

        # --- Alternative: Join with separator for fast comparison (commented for future use) ---
        # This is fast and readable, but only safe if you know the separator does not appear in any column name.
        # separator = '|'
        # joined_header = separator.join(header_to_compare)
        # joined_expected = separator.join(expected_to_compare)
        # if joined_header == joined_expected:
        #     self.logger.info("Column validation passed using joined string comparison.")
        #     return True, None

        # --- Detailed comparison (commented for future use) ---
        # mismatches = []
        # for i, (expected, actual) in enumerate(zip(expected_to_compare, header_to_compare)):
        #     if expected != actual:
        #         mismatches.append(
        #             f"Column {i}: Expected '{self.expected_column_names[i]}', Found '{header[i]}'"
        #         )
        # if mismatches:
        #     mismatch_report = "Column Mismatches: " + "; ".join(mismatches[:10])
        #     if len(mismatches) > 10:
        #         mismatch_report += f"; ... and {len(mismatches) - 10} more mismatches"
        #     self.logger.info(f"[Bad] {header}: {mismatch_report}")
        #     return False, mismatch_report

        # --- Fallback: column count mismatch ---
        if len(header_to_compare) != len(expected_to_compare):
            suffix = " (excluding output column)" if self.mode == 'training' else ""
            msg = (
                f"Column count mismatch{suffix}: "
                f"Expected {len(expected_to_compare)} columns, got {len(header_to_compare)}"
            )
            self.logger.info(f"[Bad] {header}: {msg}")
            return False, msg

        # If reached here, columns do not match but no count mismatch (should not happen)
        self.logger.info(f"[Bad] {header}: Columns do not match schema (hash and count both failed).")
        return False, "Columns do not match schema (hash and count both failed)."

    def _get_schema_types(self) -> List[str]:
        """Return the list of expected types from the schema file."""
        try:
            import orjson as json_parser
        except ImportError:
            import json as json_parser
        with open(self.config.schema_file, 'r', encoding='utf-8') as f:
            schema = json_parser.loads(f.read())
        return [v.lower() for v in schema['columns'].values()]

    def validate_dtypes(self, row: List[str], header: List[str], types: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate the data types of a row against the schema."""
        for idx, (value, dtype) in enumerate(zip(row, types)):
            if value == "" or value is None:
                continue  # Allow missing values, or handle as needed
            try:
                if dtype in ("float", "double"):
                    float(value)
                elif dtype in ("int", "integer"):
                    int(value)
                elif dtype in ("varchar", "string"):
                    str(value)
                # Add more types as needed
            except Exception:
                return False, f"Column '{header[idx]}' expects {dtype}, got '{value}'"
        return True, None

    def _process_file(self, entry) -> None:
        try:
            if not self.regex.match(entry.name):
                self._reject_file(entry, f"Invalid filename format. Expected pattern: {self.config.filename_pattern}")
                return

            # Only hash after filename is valid
            file_sha256 = file_hash(Path(entry.path))
            self.logger.info(f"Processing file: {entry.name} | SHA256: {file_sha256}")

            with open(entry.path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                header = next(reader)
                valid, err = self.validate_columns(header)
                if not valid:
                    self._reject_file(entry, err)
                    return

                types = self._get_schema_types()
                data_rows = []
                for row_num, row in enumerate(reader, start=2):
                    if len(row) < len(types):
                        row += [""] * (len(types) - len(row))
                    valid, err = self.validate_dtypes(row, header, types)
                    if not valid:
                        self._reject_file(entry, f"Dtype error at row {row_num}: {err}")
                        return
                    data_rows.append(row)

                # --- Null column check ---
                # If any column has all values as NULL/missing, reject the file
                if data_rows:  # Only check if there is at least one data row
                    columns = list(zip(*data_rows))
                    for idx, col in enumerate(columns):
                        if all(cell == "" or cell is None for cell in col):
                            self.logger.info(f"[Bad] {entry.name}: Column '{header[idx]}' contains only NULL/missing values")
                            self._reject_file(entry, f"Column '{header[idx]}' contains only NULL/missing values")
                            return

                self._accept_file(entry)
        except Exception as e:
            self._reject_file(entry, f"Error processing file: {str(e)}")
        finally:
            elapsed = (datetime.datetime.now() - start).total_seconds()
            self.logger.debug(f"[Profiling] {entry.name}: {elapsed:.4f}s")

    def _accept_file(self, entry) -> None:
        try:
            # Add hash or timestamp to filename to avoid overwrite
            base, ext = os.path.splitext(entry.name)
            unique_name = f"{base}_{file_hash(Path(entry.path))[:8]}{ext}"
            dest = self.good_dir / unique_name
            shutil.copy2(entry.path, dest)
            with self._counter_lock:
                self._good_files += 1
            self.logger.info(f"[Good] {unique_name}")
        except Exception as e:
            self.logger.error(f"Failed to accept {entry.name}: {e}", exc_info=True)

    def _reject_file(self, entry, reason: str) -> None:
        try:
            shutil.copy2(entry.path, self.bad_dir / entry.name)
            with self._counter_lock:
                self._bad_files += 1
            self._bad_file_reasons.append((entry.name, reason))  # <-- Track reason
            self.logger.info(f"[Bad] {entry.name}: {reason}")

            error_file = self.bad_dir / f"{entry.name}.error"
            with open(error_file, 'w') as f:
                f.write(f"Error: {reason}\n")
                f.write(f"Timestamp: {datetime.datetime.now()}\n")
                f.write(f"Filename Pattern: {self.config.filename_pattern}\n")
                f.write(f"Mode: {self.mode}\n")
                # Do not write schema file path to error file for privacy
                try:
                    with open(entry.path, 'r') as csv_file:
                        header = next(csv.reader(csv_file))
                        f.write(f"Actual Columns: {len(header)}\n")
                        f.write("First 5 columns: " + ", ".join(header[:5]) + "\n")
                        f.write("Last 5 columns: " + ", ".join(header[-5:] if len(header) >= 5 else header) + "\n")
                except Exception as e:
                    f.write(f"Could not read file header: {str(e)}\n")
        except Exception as e:
            self.logger.error(f"Failed to reject {entry.name}: {e}", exc_info=True)

    def validate_files_parallel(self) -> None:
        try:
            if not self.input_dir.exists():
                self.logger.error(f"Input directory does not exist: {self.input_dir}")
                raise RuntimeError(f"Input directory does not exist: {self.input_dir}")
            entries = [e for e in os.scandir(self.input_dir) if e.is_file() and e.name.endswith(CSV_EXTENSION)]
        except Exception as e:
            self.logger.error(f"Failed to scan input directory: {e}", exc_info=True)
            raise RuntimeError(str(e))

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
            return

        pct = (self._good_files / total) * 100
        msg = (
            f"\n{'='*60}\n"
            f"{self.mode.capitalize()} validation complete:\n"
            f"Total files processed: {total}\n"
            f"Good files: {self._good_files} ({pct:.1f}%)\n"
            f"Bad files: {self._bad_files} ({100-pct:.1f}%)\n"
            f"{'='*60}"
        )
        self.logger.info(msg)

        # Log rejected files and reasons
        if self._bad_file_reasons:
            self.logger.info("Rejected files and reasons:")
            for fname, reason in self._bad_file_reasons:
                self.logger.info(f"  {fname}: {reason}")

CSV_EXTENSION = os.getenv("CSV_EXTENSION", ".csv")

def file_hash(path: Path, algo: str = "sha256") -> str:
    """Compute the hash of a file using the specified algorithm."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    try:
        logging.info("Starting validation process...")

        logging.info("Step 1: Running Training Validation...")
        training_validator = FileValidator(TRAINING_CONFIG)
        training_validator.setup_directories()
        training_validator.validate_files_parallel()
        training_validator.summary()
    except Exception as e:
        logging.error(f"Training validation failed: {e}", exc_info=True)
        raise RuntimeError(str(e))

    try:
        logging.info("Step 2: Running Prediction Validation...")
        prediction_validator = FileValidator(PREDICTION_CONFIG)
        prediction_validator.setup_directories()
        prediction_validator.validate_files_parallel()
        prediction_validator.summary()
    except Exception as e:
        logging.error(f"Prediction validation failed: {e}", exc_info=True)
        raise RuntimeError(str(e))

    logging.info("Validation process complete!")

if __name__ == "__main__":
    main()