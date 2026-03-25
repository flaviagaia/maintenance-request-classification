from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ASSETS_DIR = BASE_DIR / "assets"

RAW_DATA_PATH = RAW_DIR / "nyc_311_maintenance_sample.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "maintenance_requests_processed.csv"
METRICS_PATH = PROCESSED_DIR / "model_metrics.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "validation_predictions.csv"
SUMMARY_PATH = PROCESSED_DIR / "summary.json"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
CONFUSION_MATRIX_PATH = ASSETS_DIR / "confusion_matrix.png"

DOWNLOAD_URL = (
    "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?"
    "$select=created_date,agency,complaint_type,descriptor,borough,incident_zip,latitude,longitude,"
    "location_type,street_name,status,resolution_description&"
    "$where=complaint_type%20in(%27Street%20Condition%27,%27Street%20Light%20Condition%27,"
    "%27Sidewalk%20Condition%27,%27Sewer%27,%27Water%20System%27,%27Missed%20Collection%27,"
    "%27Damaged%20Tree%27,%27Root/Sewer/Sidewalk%20Condition%27)&$limit=12000"
)

GROUP_MAPPING = {
    "Street Condition": "pavement_surface",
    "Sidewalk Condition": "pedestrian_infrastructure",
    "Street Light Condition": "lighting",
    "Water System": "water_network",
    "Sewer": "water_network",
    "Root/Sewer/Sidewalk Condition": "water_network",
    "Missed Collection": "sanitation",
    "Damaged Tree": "urban_forestry",
}
