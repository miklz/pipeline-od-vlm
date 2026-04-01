"""
Vision Language Model Pipeline.

This module defines the Kedro pipeline for vision-language model inference.
It provides a modular, extensible pipeline that supports multiple vision models
through a plugin-style architecture.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_vision_model,
    run_vlm_on_annotated_images,
    save_hf_od_vlm_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the vision language model pipeline.

    This pipeline supports multiple workflows:
    1. Single image inference
    2. Batch inference from dataset
    3. Template-based inference

    Returns:
        Kedro Pipeline instance
    """
    return pipeline(
        [
            # Node 1: Load the vision model
            node(
                func=load_vision_model,
                inputs={
                    "model_id": "params:vlm_model_id",
                    "parameters": "params:vlm_parameters",
                },
                outputs="vlm_model",
                name="load_vlm_model",
                tags=["vlm", "model_loading"],
            ),
            # Node 2: Run VLM inference on annotated images
            node(
                func=run_vlm_on_annotated_images,
                inputs={
                    "vlm_model": "vlm_model",
                    "hf_od_annotated_df": "od_annotated_images",
                    "prompt": "params:vlm_prompt",
                    "image_column": "params:vlm_image_column",
                    "inference_params": "params:vlm_inference_params",
                },
                outputs="vlm_results_df",
                name="run_vlm_on_annotated_images",
                tags=["vlm", "inference"],
            ),
            # Node 3: Save VLM results
            node(
                func=save_hf_od_vlm_results,
                inputs={
                    "hf_vlm_results_df": "vlm_results_df",
                    "output_path": "params:vlm_output_path",
                    "image_column": "params:vlm_image_column",
                },
                outputs="hf_od_vlm_summary",
                name="save_vlm_results",
                tags=["vlm", "output"],
            ),
        ],
        tags=["vlm_pipeline"],
    )
