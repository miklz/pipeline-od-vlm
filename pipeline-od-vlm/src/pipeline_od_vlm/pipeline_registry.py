"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from pipeline_od_vlm.pipelines import (
    download_dataset,
    download_model,
)
from pipeline_od_vlm.pipelines.object_detection.pipeline import (
    create_pipeline as object_detection_model_pipeline,
)
from pipeline_od_vlm.pipelines.vision_language_model.pipeline import (
    create_pipeline as vision_language_model_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    download_pipe = download_dataset.create_pipeline()
    download_model_pipe = download_model.create_pipeline()
    od_pipe = object_detection_model_pipeline()
    vlm_pipe = vision_language_model_pipeline()

    return {
        "__default__": download_pipe + download_model_pipe + od_pipe + vlm_pipe,
        "download_dataset": download_pipe,
        "download_model": download_model_pipe,
        "object_detection": download_pipe + download_model_pipe + od_pipe,
        "vision_language_model": download_pipe
        + download_model_pipe
        + od_pipe
        + vlm_pipe,
    }
