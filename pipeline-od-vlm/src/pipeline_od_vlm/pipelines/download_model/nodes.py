"""
Nodes for the download_model pipeline.

This module provides a node to download a model artifact from an MLflow
run using a configuration dictionary that contains the run_id, artifact
path, and model flavor.
"""

import logging
from typing import Any

from mlflow.artifacts import download_artifacts

logger = logging.getLogger(__name__)


def download_model_from_mlflow(model_config: dict[str, Any]) -> Any:
    """
    Download a model artifact from MLflow using a configuration dictionary.

    The dictionary must contain the following keys:

    - ``mlflow_run_id``      : The MLflow run ID from which to download the model.
    - ``mlflow_artifact_path``: The artifact path within the run
      (e.g. ``"model"`` or ``"checkpoints/best_model"``).
    - ``mlflow_flavor``      : The MLflow model flavor to use for loading
      (e.g. ``"mlflow.pyfunc"``, ``"mlflow.pytorch"``, ``"mlflow.sklearn"``).
      Defaults to ``"mlflow.pyfunc"`` when omitted.

    This design allows multiple models to be declared in
    ``parameters_download_model.yml`` as separate top-level keys, each
    holding the same sub-structure, and then referenced individually in
    the pipeline as ``params:<model_key>``.

    Args:
        model_config: Dictionary with keys ``mlflow_run_id``,
            ``mlflow_artifact_path``, and optionally ``mlflow_flavor``.

    Returns:
        The loaded MLflow model object.

    Raises:
        KeyError: If ``mlflow_run_id`` or ``mlflow_artifact_path`` are missing.
        ValueError: If ``mlflow_flavor`` is not a recognised MLflow flavor.
    """
    run_id = model_config["mlflow_run_id"]
    artifact_path = model_config["mlflow_artifact_path"]
    flavor = model_config.get("mlflow_flavor", "mlflow.pyfunc")

    logger.info(
        f"Downloading model from MLflow — run_id={run_id!r}, "
        f"artifact_path={artifact_path!r}, flavor={flavor!r}"
    )
    local_path = download_artifacts(run_id=run_id, artifact_path=artifact_path)

    logger.info("Model downloaded successfully from MLflow.")
    return local_path
