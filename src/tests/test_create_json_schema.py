import os
os.environ["BASE_DIR"] = "/tmp"
os.environ["SCHEMA_LOGS_DIR"] = "/tmp"
os.environ["SCHEMA_LOG_FILE"] = "/tmp/schema_creation.log"

import json
import logging
from pathlib import Path
import pytest

from src.create_json_schema import SchemaCreator

@pytest.fixture
def temp_schema_dir(tmp_path):
    # Use a temporary directory for schema output
    return tmp_path / "schema"

@pytest.fixture
def logger():
    # Use a dummy logger for testing
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger

def test_generate_training_schema(temp_schema_dir, logger):
    temp_schema_dir.mkdir(parents=True, exist_ok=True)
    creator = SchemaCreator(temp_schema_dir, logger)
    output_path = creator.generate_schema(
        filename="test_training_schema.json",
        num_columns=10,
        include_output=True
    )
    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert data["NumberOfColumns"] == 10
    assert "Output" in data["columns"]
    assert len(data["columns"]) == 10

def test_generate_prediction_schema(temp_schema_dir, logger):
    temp_schema_dir.mkdir(parents=True, exist_ok=True)
    creator = SchemaCreator(temp_schema_dir, logger)
    output_path = creator.generate_schema(
        filename="test_prediction_schema.json",
        num_columns=9,
        include_output=False
    )
    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert data["NumberOfColumns"] == 9
    assert "Output" not in data["columns"]
    assert len(data["columns"]) == 9

def test_invalid_schema_dir(logger):
    # Pass an invalid directory path to SchemaCreator
    with pytest.raises(Exception):
        SchemaCreator("/invalid/path/<>", logger)