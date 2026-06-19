from pathlib import Path
import re

import mlflow
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_ARTIFACT_ROOT = BASE_DIR / "mlartifacts"
DEFAULT_EXPERIMENT_NAME = "Heart Disease Prediction"
EXPERIMENT_DESCRIPTION = (
    "Binary heart disease prediction experiment using the UCI Heart Disease dataset. "
    "Tracks preprocessing, feature selection, and classifier comparison runs."
)
EXPERIMENT_TAGS = {
    "project": "CardioCare",
    "task": "binary_classification",
    "dataset": "UCI Heart Disease",
    "primary_metric": "recall",
}


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "default_experiment"


def configure_mlflow() -> str:
    MLFLOW_ARTIFACT_ROOT.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MLFLOW_TRACKING_URI


def ensure_experiment(name: str = DEFAULT_EXPERIMENT_NAME):
    configure_mlflow()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        _set_experiment_metadata(client, experiment.experiment_id)
        return client.get_experiment(experiment.experiment_id)

    artifact_dir = MLFLOW_ARTIFACT_ROOT / _slugify(name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = client.create_experiment(
        name=name,
        artifact_location=artifact_dir.resolve().as_uri(),
    )
    _set_experiment_metadata(client, experiment_id)
    return client.get_experiment(experiment_id)


def _set_experiment_metadata(client: MlflowClient, experiment_id: str) -> None:
    client.set_experiment_tag(experiment_id, "mlflow.note.content", EXPERIMENT_DESCRIPTION)
    for key, value in EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment_id, key, value)
