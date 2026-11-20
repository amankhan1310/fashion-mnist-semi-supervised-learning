"""Experiment constants and storage locations. Values carried over from the Jury 2 notebook unless noted."""
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_JSON = RESULTS_DIR / "results.json"
PREDICTIONS_NPZ = RESULTS_DIR / "test_predictions.npz"
LOG_DIR = ROOT / "logs"
TEMP_DIR = ROOT / "temp"
JOBLIB_TEMP_DIR = TEMP_DIR / "joblib"
MPL_CACHE_DIR = TEMP_DIR / "matplotlib"


def configure_storage():
    """Keep every project file inside the project folder (on D:), including temp and cache files.

    Call before importing joblib/sklearn/matplotlib. Environment variables are inherited by
    joblib/loky worker processes, so parallel workers use the same locations.
    """
    for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR, LOG_DIR, TEMP_DIR, JOBLIB_TEMP_DIR, MPL_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for var in ("TMP", "TEMP", "TMPDIR"):
        os.environ[var] = str(TEMP_DIR)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(JOBLIB_TEMP_DIR)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
    tempfile.tempdir = str(TEMP_DIR)        # in case tempfile was already initialised

RANDOM_STATE = 42
MAIN_LABEL_FRACTION = 0.10                    # 10% labelled / 90% unlabelled (-1)
LABEL_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]
ROBUSTNESS_SEEDS = [42, 123, 2026]
CV_FOLDS = 5
PCA_COMPONENTS = 50                           # feature space for the LabelSpreading graph

# Hyperparameter grids (LR and LS grids identical to the notebook; RF is new for this jury)
LR_C_GRID = [0.01, 0.1, 1.0]
LR_FIXED = dict(solver="lbfgs", max_iter=2000)
LS_K_GRID = [7, 10, 15, 20]
LS_ALPHA_GRID = [0.2, 0.5, 0.8]
LS_FIXED = dict(kernel="knn", max_iter=1000)
RF_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20],
    "max_features": ["sqrt", "log2"],
}

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MODEL_NAMES = {
    "label_spreading": "LabelSpreading (semi-supervised)",
    "logistic_regression": "Logistic Regression (supervised)",
    "random_forest": "Random Forest (supervised)",
}
