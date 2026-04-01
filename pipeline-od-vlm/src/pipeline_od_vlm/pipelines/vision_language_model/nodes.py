"""
Kedro nodes for vision-language model pipeline.

This module provides modular, reusable nodes for vision model inference
within the Kedro pipeline framework. The nodes are designed to work with
multiple vision models through a plugin-style architecture.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from pipeline_od_vlm.common.base import BaseVisionModel, ModelConfig
from pipeline_od_vlm.common.factory import ModelFactory, get_model_for_name

from .utils import (
    load_image,
    load_images,
)

logger = logging.getLogger(__name__)


def load_vision_model(model_id: str, parameters: dict[str, Any]) -> BaseVisionModel:
    """
    Load a vision model based on configuration.

    This node loads and initializes a vision model using the factory pattern.
    It supports multiple model types through the model registry.

    Args:
        model_id: Identifier for the model to load (e.g., 'qwen3-vl', 'llava-next')
        parameters: Dictionary containing model parameters from Kedro config

    Returns:
        Loaded vision model instance

    Example:
        >>> model = load_vision_model("qwen3-vl", params)
    """
    logger.info(f"Loading vision model: {model_id}")

    # Get model-specific parameters
    model_params = parameters.get(model_id, {})

    if not model_params:
        raise ValueError(
            f"No parameters found for model '{model_id}'. "
            f"Please check your parameters configuration."
        )

    # Create and load the model
    model = ModelFactory.create_model_from_dict(
        model_id=model_id, config_dict=model_params, load_immediately=True
    )

    logger.info(f"Successfully loaded model: {model_id}")
    return model


def run_inference(
    model: BaseVisionModel,
    image_data: str | Path | Image.Image | pd.DataFrame,
    prompt: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run inference on image(s) using a vision model.

    This node performs inference using a loaded vision model. It supports
    single images, multiple images, and batch processing from DataFrames.

    Args:
        model: Loaded vision model instance
        image_data: Input image(s) - can be path, URL, PIL Image, or DataFrame
        prompt: Text prompt for the model
        parameters: Optional inference parameters (max_tokens, temperature, etc.)

    Returns:
        Dictionary containing inference results and metadata

    Example:
        >>> results = run_inference(model, "path/to/image.jpg", "Describe this image")
    """
    logger.info("Running vision model inference")

    params = parameters or {}

    # Handle different input types
    if isinstance(image_data, pd.DataFrame):
        # Batch processing from DataFrame
        results = _run_batch_inference(model, image_data, prompt, params)
    else:
        # Single image inference
        results = _run_single_inference(model, image_data, prompt, params)

    logger.info("Inference completed successfully")
    return results


def _run_single_inference(
    model: BaseVisionModel,
    image: str | Path | Image.Image,
    prompt: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Run inference on a single image.

    Args:
        model: Vision model instance
        image: Input image
        prompt: Text prompt
        params: Inference parameters

    Returns:
        Inference results
    """
    # Load image if needed
    if not isinstance(image, Image.Image):
        image = load_image(image)

    # Run inference
    output = model.inference(image, prompt, **params)

    # Format results
    return {"text": output.text, "metadata": output.metadata, "success": True}


def _run_batch_inference(
    model: BaseVisionModel, df: pd.DataFrame, prompt: str, params: dict[str, Any]
) -> dict[str, Any]:
    """
    Run inference on multiple images from a DataFrame.

    Args:
        model: Vision model instance
        df: DataFrame containing image paths or data
        prompt: Text prompt
        params: Inference parameters

    Returns:
        Batch inference results
    """
    results = []

    # Assume DataFrame has an 'image_path' or 'image' column
    image_column = "image_path" if "image_path" in df.columns else "image"

    for idx, row in df.iterrows():
        try:
            image_path = row[image_column]
            output = model.inference(image_path, prompt, **params)

            results.append(
                {
                    "index": idx,
                    "text": output.text,
                    "metadata": output.metadata,
                    "success": True,
                }
            )
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {e}")
            results.append({"index": idx, "error": str(e), "success": False})

    return {
        "results": results,
        "total": len(results),
        "successful": sum(1 for r in results if r["success"]),
    }


def batch_inference_from_dataset(
    model: BaseVisionModel,
    dataset: pd.DataFrame,
    prompt_column: str = "prompt",
    image_column: str = "image_path",
    parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run batch inference on a dataset with per-row prompts.

    This node processes a dataset where each row has its own image and prompt,
    useful for evaluation or large-scale processing.

    Args:
        model: Loaded vision model instance
        dataset: DataFrame with images and prompts
        prompt_column: Name of column containing prompts
        image_column: Name of column containing image paths
        parameters: Optional inference parameters

    Returns:
        DataFrame with original data plus inference results

    Example:
        >>> results_df = batch_inference_from_dataset(model, eval_dataset)
    """
    logger.info(f"Running batch inference on {len(dataset)} samples")

    params = parameters or {}
    results = []

    for idx, row in dataset.iterrows():
        try:
            image_path = row[image_column]
            prompt = row[prompt_column]

            output = model.inference(image_path, prompt, **params)

            results.append(
                {**row.to_dict(), "prediction": output.text, "success": True}
            )
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            results.append(
                {**row.to_dict(), "prediction": None, "error": str(e), "success": False}
            )

    results_df = pd.DataFrame(results)
    logger.info(
        f"Batch inference completed: {results_df['success'].sum()}/{len(results_df)} successful"
    )

    return results_df


def save_inference_results(results: dict[str, Any], output_path: str | Path) -> None:
    """
    Save inference results to disk.

    This node saves inference results in a structured format for later use.

    Args:
        results: Inference results dictionary
        output_path: Path to save results

    Example:
        >>> save_inference_results(results, "data/07_model_output/results.json")
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved inference results to {output_path}")


def preprocess_images(
    image_paths: list[str | Path],
    resize: tuple | None = None,
    convert_mode: str | None = None,
) -> list[Image.Image]:
    """
    Preprocess images before inference.

    This node handles common preprocessing tasks like resizing and format conversion.

    Args:
        image_paths: List of image paths
        resize: Optional (width, height) tuple for resizing
        convert_mode: Optional PIL mode to convert to (e.g., 'RGB')

    Returns:
        List of preprocessed PIL Images

    Example:
        >>> images = preprocess_images(paths, resize=(224, 224), convert_mode="RGB")
    """
    logger.info(f"Preprocessing {len(image_paths)} images")

    images = load_images(image_paths)

    processed = []
    for img in images:
        if convert_mode and img.mode != convert_mode:
            img = img.convert(convert_mode)

        if resize:
            img = img.resize(resize)

        processed.append(img)

    logger.info("Image preprocessing completed")
    return processed


def create_model_config(model_name: str, parameters: dict[str, Any]) -> ModelConfig:
    """
    Create a ModelConfig from parameters.

    This node creates a structured configuration object from Kedro parameters.

    Args:
        model_name: Name of the model
        parameters: Parameters dictionary

    Returns:
        ModelConfig instance

    Example:
        >>> config = create_model_config("Qwen/Qwen3-VL-8B-Instruct", params)
    """

    # Extract model-specific parameters
    model_id = get_model_for_name(model_name)
    model_params = parameters.get(model_id, {})

    # Convert dtype string to torch.dtype if needed
    dtype = model_params.get("dtype", "bfloat16")
    if isinstance(dtype, str):
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype, torch.bfloat16)

    config = ModelConfig(
        model_name=model_params.get("model_name", model_name),
        model_folder_name=model_params.get("model_folder_name", ""),
        dtype=dtype,
        device_map=model_params.get("device_map", "auto"),
        attn_implementation=model_params.get("attn_implementation"),
        max_new_tokens=model_params.get("max_new_tokens", 128),
        temperature=model_params.get("temperature", 0.7),
        top_p=model_params.get("top_p", 0.9),
        additional_params=model_params.get("additional_params"),
    )

    return config


def run_vlm_on_annotated_images(
    vlm_model: BaseVisionModel,
    hf_od_annotated_df: pd.DataFrame,
    prompt: str,
    image_column: str = "image",
    inference_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run VLM inference on annotated images (with bounding boxes drawn).

    The VLM receives the annotated image (with bounding boxes) as visual input.
    The prompt does not need to include detection metadata since the model can
    see the boxes directly.

    Args:
        vlm_model: Loaded vision language model
        hf_od_annotated_df: DataFrame with annotated images (bounding boxes drawn)
        prompt: Prompt for VLM inference
        image_column: Name of the column containing annotated PIL images
        inference_params: Optional inference parameters

    Returns:
        DataFrame with original data plus VLM inference results:
        - vlm_response: text response from VLM
        - vlm_success: whether inference succeeded
    """
    logger.info(f"Running VLM inference on {len(hf_od_annotated_df)} annotated images")

    params = inference_params or {}
    results = []

    for idx, row in hf_od_annotated_df.iterrows():
        try:
            image = row[image_column]

            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            elif image.mode != "RGB":
                image = image.convert("RGB")

            # Run VLM inference on the annotated image
            output = vlm_model.inference(image, prompt, **params)

            record = {k: v for k, v in row.items() if k != image_column}
            record[image_column] = image
            record["vlm_response"] = output.text
            record["vlm_success"] = True

            results.append(record)

            logger.debug(f"Image {idx}: VLM inference completed")

        except Exception as e:
            logger.error(f"Error running VLM on image at index {idx}: {e}")
            record = {k: v for k, v in row.items() if k != image_column}
            record[image_column] = row.get(image_column)
            record["vlm_response"] = None
            record["vlm_success"] = False
            record["vlm_error"] = str(e)
            results.append(record)

    results_df = pd.DataFrame(results)
    successful = results_df["vlm_success"].sum()
    logger.info(
        f"VLM completed: {successful}/{len(results_df)} images processed successfully"
    )

    return results_df


def save_hf_od_vlm_results(
    hf_vlm_results_df: pd.DataFrame,
    output_path: str,
    image_column: str = "image",
) -> dict[str, Any]:
    """
    Save the combined OD + VLM results to a parquet file via catalog.

    Saves the final results, excluding PIL Image objects which cannot be
    serialized to parquet. Returns a summary dict and writes the parquet
    file to the path specified.

    Args:
        hf_vlm_results_df: DataFrame with OD and VLM results
        output_path: Path to save the parquet file
        image_column: Name of the image column to exclude from saving

    Returns:
        Summary statistics dictionary (hf_od_vlm_summary)
    """

    logger.info(f"Saving results to {output_path}")

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop PIL Image column for serialization
    save_df = hf_vlm_results_df.drop(columns=[image_column], errors="ignore")

    # Convert list/dict columns to JSON strings for parquet compatibility
    for col in save_df.columns:
        if save_df[col].dtype == object:
            try:
                save_df[col] = save_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
            except Exception:
                pass

    # Save to parquet
    save_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(save_df)} records to {output_path}")

    # Create summary
    summary = {
        "total_images": len(hf_vlm_results_df),
        "od_successful": int(
            hf_vlm_results_df.get(
                "od_success", pd.Series([True] * len(hf_vlm_results_df))
            ).sum()
        ),
        "vlm_successful": int(
            hf_vlm_results_df.get(
                "vlm_success", pd.Series([True] * len(hf_vlm_results_df))
            ).sum()
        ),
        "total_detections": int(
            hf_vlm_results_df.get(
                "num_detections", pd.Series([0] * len(hf_vlm_results_df))
            ).sum()
        ),
        "output_path": str(output_path),
    }

    logger.info(f"Pipeline summary: {summary}")
    return summary
