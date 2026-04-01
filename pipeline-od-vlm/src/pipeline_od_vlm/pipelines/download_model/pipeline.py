"""
Download model pipeline definition.

This pipeline downloads a model artifact from an MLflow run identified
by a specific run_id and stores it as a MemoryDataset for downstream
pipelines (object_detection, vision_language_model).

Each model to be downloaded is declared as a top-level key in
``parameters_download_model.yml`` with the sub-keys:
  - ``mlflow_run_id``
  - ``mlflow_artifact_path``
  - ``mlflow_flavor``

Example YAML entry::

    od_rtdetrv2_mlflow_model:
      mlflow_run_id: "232629237ef34e4eacda191e443b2598"
      mlflow_artifact_path: "model"
      mlflow_flavor: "mlflow.pytorch"

The corresponding pipeline node receives ``params:od_rtdetrv2_mlflow_model``
as its single input and outputs the local filesystem path to the downloaded
artifact as the ``model_path`` MemoryDataset for downstream pipelines.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_model_from_mlflow


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the download_model pipeline.

    Downloads a model from MLflow using the configuration dict stored under
    ``params:od_rtdetrv2_mlflow_model`` and outputs the local filesystem path
    to the downloaded artifact as the ``model_path`` MemoryDataset for
    downstream pipelines.

    To add more models, duplicate the node block and change the parameter
    key and output dataset name accordingly.

    Returns:
        Kedro Pipeline for downloading MLflow models.
    """
    return pipeline(
        [
            node(
                func=download_model_from_mlflow,
                inputs="params:od_rtdetrv2_mlflow_model",
                outputs="model_path",
                name="download_od_rtdetrv2_mlflow_model",
                tags=["download_model"],
            ),
        ],
        tags=["download_model"],
    )
