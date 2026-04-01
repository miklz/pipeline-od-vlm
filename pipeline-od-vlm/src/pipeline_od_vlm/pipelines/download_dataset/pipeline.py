"""
Download dataset pipeline definition.

This pipeline downloads a HuggingFace dataset and stores it as a
MemoryDataset for downstream pipelines (object_detection, vision_language_model).
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_hf_dataset


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the download_dataset pipeline.

    Downloads the HuggingFace dataset and outputs it as a MemoryDataset
    (maritime_sar_optical_descriptions) for use by the object_detection pipeline.

    Returns:
        Kedro Pipeline for downloading the HF dataset
    """
    return pipeline(
        [
            node(
                func=download_hf_dataset,
                inputs={
                    "dataset_name": "params:hf_dataset_name",
                    "split": "params:hf_dataset_split",
                    "max_samples": "params:hf_max_samples",
                },
                outputs="maritime_sar_optical_descriptions",
                name="download_hf_dataset",
                tags=["download_dataset"],
            ),
        ],
        tags=["download_dataset"],
    )
