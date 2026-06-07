"""Project-wide configuration and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

CREDITCARD_FILENAME = "creditcard.csv"
FRAUD_DATA_FILENAME = "Fraud_Data.csv"
IP_COUNTRY_FILENAME = "IpAddress_to_Country.csv"
