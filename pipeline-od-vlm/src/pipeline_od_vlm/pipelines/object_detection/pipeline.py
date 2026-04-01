"""
Object detection pipeline definition.

This module defines the Kedro pipeline for object detection tasks.
The pipeline is modular and supports multiple object detection models.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    detect_objects_in_image,
    draw_detections_on_dataset,
    load_object_detection_model,
    save_detection_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the object detection pipeline.

    This pipeline supports multiple workflows:
    1. Single image detection
    2. Batch detection on datasets
    3. Detection with visualization (bounding boxes)
    4. Filtered detection by class

    Returns:
        Kedro Pipeline for object detection
    """
    return pipeline(
        [
            # Load object detection model
            # model_path is the local filesystem path returned by the download_model
            # pipeline after downloading the artifact from MLflow.
            # When od_model_id == "rt-deterv2" the model is loaded from this path,
            # bypassing the HuggingFace download.
            node(
                func=load_object_detection_model,
                inputs=[
                    "params:od_model_id",
                    "params:od_parameters",
                    "model_path",
                ],
                outputs="od_model",
                name="load_od_model",
                tags=["object_detection", "model_loading"],
            ),
            # Single image detection
            node(
                func=detect_objects_in_image,
                inputs=[
                    "od_model",
                    "maritime_sar_optical_descriptions",
                    "params:od_confidence_threshold",
                    "params:od_detection_params",
                ],
                outputs="od_results",
                name="detect_objects_single",
                tags=["object_detection", "single_image"],
            ),
            # Draw bounding boxes on dataset (batch visualization)
            node(
                func=draw_detections_on_dataset,
                inputs=[
                    "maritime_sar_optical_descriptions",
                    "od_results",
                    "params:od_batch_output_dir",
                    "params:od_box_style",
                ],
                outputs="od_annotated_images",
                name="draw_detections_batch",
                tags=["object_detection", "visualization", "batch"],
            ),
            # Save detection results
            node(
                func=save_detection_results,
                inputs=["od_results", "params:od_results_output_path"],
                outputs=None,
                name="save_detection_results",
                tags=["object_detection", "saving"],
            ),
            # Save annotated images metadata
            node(
                func=save_detection_results,
                inputs=["od_annotated_images", "params:od_annotated_output_path"],
                outputs=None,
                name="save_annotated_metadata",
                tags=["object_detection", "saving"],
            ),
        ],
        tags="object_detection",
    )
