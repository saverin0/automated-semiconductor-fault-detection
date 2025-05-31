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
import time

# --- Load environment variables ---
load_dotenv()

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
        if self.mode != 'training':
            raise ValueError(f"Invalid mode: {self.mode}. Only 'training' mode is supported.")

def setup_environment():
    """Setup environment variables and paths."""
    BASE_DIR = Path(os.getenv("BASE_DIR", "/workspaces/automated-semiconductor-fault-detection"))

    # Fallback logging
    FALLBACK_LOGS_DIR = validate_env_path(os.getenv("FALLBACK_LOGS_DIR"), BASE_DIR)
    FALLBACK_LOG_FILE = validate_env_path(os.getenv("FALLBACK_LOG_FILE"), FALLBACK_LOGS_DIR)
    ensure_path_exists(FALLBACK_LOGS_DIR)
    ensure_path_exists(FALLBACK_LOG_FILE, is_file=True)

    logging.basicConfig(
        filename=FALLBACK_LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logging.info("Fallback logging initialized.")

    # Training configuration
    try:
        config = ValidatorConfig(
            input_dir=str(validate_env_path(os.environ['TRAINING_INPUT_DIR'], BASE_DIR)),
            good_dir=str(validate_env_path(os.environ['TRAINING_GOOD_DIR'], BASE_DIR)),
            bad_dir=str(validate_env_path(os.environ['TRAINING_BAD_DIR'], BASE_DIR)),
            schema_file=str(validate_env_path(os.environ['TRAINING_SCHEMA_FILE'], BASE_DIR)),
            log_file=str(validate_env_path(os.environ['TRAINING_LOG_FILE'], BASE_DIR)),
            filename_pattern=os.getenv('FILENAME_PATTERN', r"^wafer_\d{8}_\d{6}\.csv$"),
            max_workers=int(os.getenv('MAX_WORKERS', '4')),
            mode="training"
        )

        # Ensure paths exist
        for path_str in [config.input_dir, config.good_dir, config.bad_dir]:
            ensure_path_exists(Path(path_str))
        ensure_path_exists(Path(config.schema_file), is_file=True)
        ensure_path_exists(Path(config.log_file), is_file=True)

        logging.info("Environment setup and config validation successful.")
        return config

    except (KeyError, ValueError) as e:
        logging.error(f"Environment setup failed: {e}", exc_info=True)
        raise RuntimeError(f"Invalid environment configuration: {e}")

def get_logger(log_file: str, mode: str) -> logging.Logger:
    """Get a logger with mode-specific configuration."""
    logger = logging.getLogger(f"validator.{mode}")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # <-- This prevents duplicate handlers
    handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def normalize_colname(name: str) -> str:
    """Normalize column names for comparison."""
    return name.lower().replace(' ', '').replace('-', '')

def hash_columns(columns: list) -> str:
    """Hash column names for fast comparison."""
    joined = '|'.join(columns)
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()

def file_hash(path: Path, algo: str = "sha256") -> str:
    """Compute file hash."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

class FileValidator:
    def __init__(self, config: ValidatorConfig):
        logging.info("Initializing FileValidator...")
        config.validate()
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.good_dir = Path(config.good_dir)
        self.bad_dir = Path(config.bad_dir)
        self.logger = get_logger(config.log_file, config.mode)
        self.regex = re.compile(config.filename_pattern)

        # Thread-safe counters
        self._counter_lock = Lock()
        self._good_files = 0
        self._bad_files = 0
        self._bad_file_reasons = []

        # Cache for file hashes to avoid recomputation
        self._file_hashes = {}

        self.logger.info(f"Validator initialized for mode: {config.mode}")
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize schema from file."""
        try:
            self.logger.info("Reading schema file...")
            self.expected_column_names = self._read_schema()
            self.expected_types = self._get_schema_types()
            self.logger.info(f"Schema loaded: {len(self.expected_column_names)} columns")
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {e}", exc_info=True)
            raise

    @lru_cache(maxsize=1)
    def _read_schema(self) -> List[str]:
        """Read and cache schema columns."""
        try:
            import orjson as json_parser
        except ImportError:
            import json as json_parser

        with open(self.config.schema_file, 'r', encoding='utf-8') as f:
            schema = json_parser.loads(f.read())

        if 'columns' not in schema:
            raise ValueError("Schema missing 'columns' section")

        return [name.strip() for name in schema['columns'].keys()]

    def _get_schema_types(self) -> List[str]:
        """Get expected data types from schema."""
        try:
            import orjson as json_parser
        except ImportError:
            import json as json_parser

        with open(self.config.schema_file, 'r', encoding='utf-8') as f:
            schema = json_parser.loads(f.read())
        return [v.lower() for v in schema['columns'].values()]

    def _get_file_hash(self, file_path: Path) -> str:
        """Get file hash with caching."""
        if file_path not in self._file_hashes:
            self._file_hashes[file_path] = file_hash(file_path)
        return self._file_hashes[file_path]

    def setup_directories(self) -> None:
        """Setup validation directories."""
        self.logger.info("Setting up directories for validation...")
        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            raise RuntimeError(f"Input directory does not exist: {self.input_dir}")

        # Clean and recreate output directories
        for path in [self.good_dir, self.bad_dir]:
            if path.exists():
                shutil.rmtree(path)
                self.logger.info(f"Removed existing directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {path}")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info("=" * 60)
        self.logger.info(f"=== {self.config.mode.upper()} Validation started at {timestamp} ===")
        self.logger.info("=" * 60)

    def validate_columns(self, header: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate CSV header against schema."""
        self.logger.debug(f"Validating columns: {header}")
        if not header:
            self.logger.warning("Header is empty")
            return False, "Header is empty"

        # Handle unnamed first column
        if header[0] == "" or header[0].lower() == "unnamed: 0":
            header[0] = "Wafer"

        # Normalize column names
        expected_norm = [normalize_colname(n) for n in self.expected_column_names]
        header_norm = [normalize_colname(n) for n in header]

        # Check last column for training mode
        if header_norm[-1] not in ("output", "good/bad"):
            self.logger.warning(f"Last column should be 'Output' or 'Good/Bad', got '{header[-1]}'")
            return False, f"Last column should be 'Output' or 'Good/Bad', got '{header[-1]}'"

        # Compare all columns except the last one using hash
        header_to_compare = header_norm[:-1]
        expected_to_compare = expected_norm[:-1]

        if hash_columns(header_to_compare) == hash_columns(expected_to_compare):
            self.logger.info("Column validation passed using hash comparison.")
            return True, None

        # Check column count
        if len(header_to_compare) != len(expected_to_compare):
            self.logger.warning(f"Column count mismatch: expected {len(expected_to_compare)}, got {len(header_to_compare)}")
            return False, f"Column count mismatch: expected {len(expected_to_compare)}, got {len(header_to_compare)}"

        self.logger.warning("Columns do not match schema")
        return False, "Columns do not match schema"

    def validate_dtypes(self, row: List[str], header: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate row data types against schema."""
        for idx, (value, dtype) in enumerate(zip(row, self.expected_types)):
            # Allow missing values - they will be handled later in the preprocessing stage
            if not value or value.strip() == "":
                continue

            try:
                if dtype in ("float", "double"):
                    float(value)
                elif dtype in ("int", "integer"):
                    int(value)
                elif dtype in ("varchar", "string"):
                    str(value)
            except ValueError:
                self.logger.warning(f"Column '{header[idx]}' expects {dtype}, got '{value}'")
                return False, f"Column '{header[idx]}' expects {dtype}, got '{value}'"

        return True, None

    def _process_file(self, entry) -> None:
        """Process a single file."""
        start_time = time.time()
        self.logger.info(f"Processing file: {entry.name}")

        try:
            # Validate filename
            if not self.regex.match(entry.name):
                self.logger.warning(f"Invalid filename format: {entry.name}")
                self._reject_file(entry, f"Invalid filename format")
                return

            # Get file hash
            file_sha256 = self._get_file_hash(Path(entry.path))
            self.logger.info(f"File SHA256: {file_sha256[:16]}...")

            # Validate file content
            with open(entry.path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)

                try:
                    header = next(reader)
                except StopIteration:
                    self.logger.warning(f"File is empty: {entry.name}")
                    self._reject_file(entry, "File is empty")
                    return

                # Validate columns
                valid, error = self.validate_columns(header)
                if not valid:
                    self.logger.warning(f"Column validation failed: {error}")
                    self._reject_file(entry, error)
                    return

                # Validate data rows
                data_rows = []
                for row_num, row in enumerate(reader, start=2):
                    # Pad short rows
                    if len(row) < len(self.expected_types):
                        row += [""] * (len(self.expected_types) - len(row))

                    valid, error = self.validate_dtypes(row, header)
                    if not valid:
                        self.logger.warning(f"Row {row_num} dtype validation failed: {error}")
                        self._reject_file(entry, f"Row {row_num}: {error}")
                        return

                    data_rows.append(row)

                # Check if file has any data
                if not data_rows:
                    self.logger.warning(f"File contains no data rows: {entry.name}")
                    self._reject_file(entry, "File contains no data rows")
                    return

                # Basic sanity check: reject only if ALL columns are completely null
                # (Following the original project's approach - let preprocessing handle missing values)
                if data_rows:
                    columns = list(zip(*data_rows))
                    completely_null_columns = []
                    for idx, col in enumerate(columns):
                        if all(not cell or cell.strip() == "" for cell in col):
                            completely_null_columns.append(header[idx])

                    # Only reject if ALL feature columns (excluding target) are completely null
                    feature_columns = header[:-1]  # Exclude target column
                    if len(completely_null_columns) >= len(feature_columns):
                        self.logger.warning(f"All feature columns are completely null: {entry.name}")
                        self._reject_file(entry, "All feature columns contain only null values")
                        return

                self._accept_file(entry, file_sha256)

        except Exception as e:
            self.logger.error(f"Processing error for {entry.name}: {str(e)}", exc_info=True)
            self._reject_file(entry, f"Processing error: {str(e)}")
        finally:
            elapsed = time.time() - start_time
            self.logger.debug(f"Processed {entry.name} in {elapsed:.3f}s")

    def _accept_file(self, entry, file_hash: str) -> None:
        """Accept a valid file."""
        try:
            base, ext = os.path.splitext(entry.name)
            unique_name = f"{base}_{file_hash[:8]}{ext}"
            dest = self.good_dir / unique_name
            shutil.copy2(entry.path, dest)

            with self._counter_lock:
                self._good_files += 1

            self.logger.info(f"[GOOD] {unique_name}")
        except Exception as e:
            self.logger.error(f"Failed to accept {entry.name}: {e}")

    def _reject_file(self, entry, reason: str) -> None:
        """Reject an invalid file."""
        try:
            shutil.copy2(entry.path, self.bad_dir / entry.name)

            with self._counter_lock:
                self._bad_files += 1
                self._bad_file_reasons.append((entry.name, reason))

            self.logger.info(f"[BAD] {entry.name}: {reason}")

            # Create error file
            error_file = self.bad_dir / f"{entry.name}.error"
            with open(error_file, 'w') as f:
                f.write(f"Error: {reason}\n")
                f.write(f"Timestamp: {datetime.datetime.now()}\n")
                f.write(f"Mode: {self.config.mode}\n")

        except Exception as e:
            self.logger.error(f"Failed to reject {entry.name}: {e}")

    def validate_files_parallel(self) -> None:
        """Validate all files in parallel."""
        try:
            csv_extension = os.getenv("CSV_EXTENSION", ".csv")
            self.logger.info(f"Scanning input directory {self.input_dir} for CSV files...")
            entries = [
                e for e in os.scandir(self.input_dir)
                if e.is_file() and e.name.endswith(csv_extension)
            ]
        except Exception as e:
            self.logger.error(f"Failed to scan input directory: {e}")
            raise

        if not entries:
            self.logger.warning(f"No CSV files found in {self.input_dir}")
            return

        self.logger.info(f"Found {len(entries)} CSV files to process")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            list(executor.map(self._process_file, entries))

    def summary(self) -> None:
        """Print validation summary."""
        total = self._good_files + self._bad_files
        if total == 0:
            self.logger.warning("No files processed")
            return

        success_rate = (self._good_files / total) * 100

        summary_msg = (
            f"\n{'=' * 60}\n"
            f"{self.config.mode.capitalize()} Validation Summary:\n"
            f"Total files: {total}\n"
            f"Valid files: {self._good_files} ({success_rate:.1f}%)\n"
            f"Invalid files: {self._bad_files} ({100 - success_rate:.1f}%)\n"
            f"Note: Missing value handling will occur during preprocessing stage\n"
            f"{'=' * 60}"
        )
        self.logger.info(summary_msg)

        # Log rejection reasons
        if self._bad_file_reasons:
            self.logger.info("Rejection reasons:")
            for filename, reason in self._bad_file_reasons:
                self.logger.info(f"  {filename}: {reason}")