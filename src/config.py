"""Project-wide configuration and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELING_OUTPUTS_DIR = REPORTS_DIR / "outputs"
CREDITCARD_OUTPUTS_DIR = MODELING_OUTPUTS_DIR / "creditcard"
MODELING_PLOTS_DIR = MODELING_OUTPUTS_DIR / "plots"
CREDITCARD_PLOTS_DIR = CREDITCARD_OUTPUTS_DIR / "plots"
MODELING_CONFUSION_DIR = MODELING_OUTPUTS_DIR / "confusion_matrices"
CREDITCARD_CONFUSION_DIR = CREDITCARD_OUTPUTS_DIR / "confusion_matrices"
SHAP_FIGURES_DIR = REPORTS_DIR / "figures"
CREDITCARD_SHAP_FIGURES_DIR = SHAP_FIGURES_DIR / "creditcard"
SHAP_REPORTS_DIR = REPORTS_DIR / "shap"
CREDITCARD_SHAP_REPORTS_DIR = SHAP_REPORTS_DIR / "creditcard"
# Legacy path kept for backward compatibility
MODELING_REPORTS_DIR = MODELING_OUTPUTS_DIR

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

CREDITCARD_FILENAME = "creditcard.csv"
CREDITCARD_CLEAN_FILENAME = "creditcard_clean.csv"
CREDITCARD_FEATURES_FILENAME = "creditcard_features.csv"
FRAUD_DATA_FILENAME = "Fraud_Data.csv"
IP_COUNTRY_FILENAME = "IpAddress_to_Country.csv"
FRAUD_DATA_GEOLOCATED_FILENAME = "fraud_data_geolocated.csv"
FRAUD_DATA_ENGINEERED_FILENAME = "fraud_data_engineered.csv"
FRAUD_DATA_FEATURES_FILENAME = "fraud_data_features.csv"
