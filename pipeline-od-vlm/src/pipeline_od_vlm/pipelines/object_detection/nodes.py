"""
Kedro nodes for object detection pipeline.

This module provides modular, reusable nodes for object detection
within the Kedro pipeline framework. The nodes are designed to work with
multiple object detection models through a plugin-style architecture.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from pipeline_od_vlm.common.base import BaseObjectDetectionModel
from pipeline_od_vlm.common.factory import ModelFactory

from .models import draw_bounding_boxes, draw_bounding_boxes_cv2

logger = logging.getLogger(__name__)


def load_object_detection_model(
    model_id: str,
    parameters: dict[str, Any],
    model_path: str | None = None,
) -> BaseObjectDetectionModel:
    """
    Load an object detection model based on configuration.

    This node loads and initializes an object detection model using the factory pattern.
    It supports multiple model types through the model registry.

    When ``model_path`` is provided (e.g. a local path returned by the
    ``download_model`` node after downloading from MLflow), the model weights are
    loaded from that directory and injected directly into the wrapper, bypassing
    the HuggingFace / remote download.

    Args:
        model_id: Identifier for the model to load (e.g., 'rt-deterv2', 'yolo')
        parameters: Dictionary containing model parameters from Kedro config
        model_path: Optional local filesystem path to a previously downloaded model
            directory (e.g. returned by ``download_model_from_mlflow``).
            When supplied the wrapper loads the model weights from this path
            instead of fetching them from HuggingFace.

    Returns:
        Loaded object detection model instance

    Example:
        >>> model = load_object_detection_model("rt-deterv2", params)
        >>> model = load_object_detection_model(
        ...     "rt-deterv2", params, "/tmp/mlflow/model"
        ... )
    """
    logger.info(f"Loading object detection model: {model_id}")

    # Get model-specific parameters
    model_params = parameters.get(model_id, {})

    if not model_params:
        raise ValueError(
            f"No parameters found for model '{model_id}'. "
            f"Please check your parameters configuration."
        )

    if model_path is not None:
        logger.info(
            f"Loading model weights from local path for '{model_id}': {model_path}"
        )
        # Create the wrapper without triggering load_model() yet
        model = ModelFactory.create_model_from_dict(
            model_id=model_id, config_dict=model_params, load_immediately=False
        )
        # Override model_name so that load_model() loads weights from the local path
        model.config.model_name = str(model_path)
        # Call load_model() to load both the model weights and the processor
        model.load_model()
    else:
        # Create and load the model normally (HuggingFace / local path)
        model = ModelFactory.create_model_from_dict(
            model_id=model_id, config_dict=model_params, load_immediately=True
        )

    logger.info(f"Successfully loaded model: {model_id}")
    return model


def detect_objects_in_image(
    model: BaseObjectDetectionModel,
    image: str | Path | Image.Image | pd.DataFrame,
    confidence_threshold: float = 0.5,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any] | pd.DataFrame:
    """
    Detect objects in a single image or a dataset of images.

    This node performs object detection on either a single image or a dataset
    of images using a loaded model. When a DataFrame is provided, it processes
    each image in the dataset and returns results as a DataFrame.

    Args:
        model: Loaded object detection model instance
        image: Input image or dataset - can be:
            - Single image: path (str/Path), URL, or PIL Image
            - Dataset: pandas DataFrame with 'image' column
        confidence_threshold: Minimum confidence for detections
        parameters: Optional detection parameters

    Returns:
        - For single image: Dictionary containing detections and metadata
        - For dataset: DataFrame with detection results for each image

    Example:
        >>> # Single image
        >>> results = detect_objects_in_image(model, "path/to/image.jpg", 0.5)
        >>> # Dataset
        >>> df = pd.DataFrame({"image": ["img1.jpg", "img2.jpg"]})
        >>> results_df = detect_objects_in_image(model, df, 0.5)
    """
    # Check if input is a DataFrame (dataset)
    if isinstance(image, pd.DataFrame):
        logger.info(f"Running object detection on dataset with {len(image)} images")
        return _detect_objects_in_dataset(
            model, image, confidence_threshold, parameters
        )

    # Single image processing
    logger.info("Running object detection on single image")

    params = parameters or {}

    # Load image based on type
    if isinstance(image, dict) and "path" in image:
        # HuggingFace Image format
        image_path = image["path"]
        image = Image.open(image_path).convert("RGB")
    elif isinstance(image, (str, Path)):
        image_path = str(image)
        image = Image.open(image).convert("RGB")
    else:
        image_path = None

    # Detect objects
    detections = model.detect_objects(
        image, confidence_threshold=confidence_threshold, **params
    )

    # Format results
    result = {
        "detections": detections,
        "num_detections": len(detections),
        "image_path": image_path,
        "confidence_threshold": confidence_threshold,
        "success": True,
    }

    logger.info(f"Detected {len(detections)} objects")
    return result


def _detect_objects_in_dataset(
    model: BaseObjectDetectionModel,
    dataset: pd.DataFrame,
    confidence_threshold: float = 0.5,
    parameters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Process a dataset of images for object detection.

    Args:
        model: Loaded object detection model instance
        dataset: DataFrame with 'image' column
        confidence_threshold: Minimum confidence for detections
        parameters: Optional detection parameters

    Returns:
        DataFrame with detection results for each image
    """
    params = parameters or {}
    results = []

    # Validate dataset has required column
    if "image" not in dataset.columns:
        raise ValueError(
            "Dataset must contain 'image' column. "
            f"Available columns: {list(dataset.columns)}"
        )

    for idx, row in dataset.iterrows():
        image = row["image"]
        try:
            # Load and process image
            # Handle different image types:
            # 1. HuggingFace Image dict: {'bytes': ..., 'path': ...}
            # 2. String/Path: file path or URL
            # 3. PIL Image object

            if isinstance(image, dict) and "path" in image:
                # HuggingFace Image format - extract the path
                image_path = image["path"]
                img = Image.open(image_path).convert("RGB")
                logger.info(f"Loaded HuggingFace image from: {image_path}")
            elif isinstance(image, (str, Path)):
                img = Image.open(image).convert("RGB")
                image_path = str(image)
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
                image_path = None
            else:
                # Fallback: try to use as-is
                img = image
                image_path = None

            detections = model.detect_objects(
                img, confidence_threshold=confidence_threshold, **params
            )

            # Store results
            result = {
                "image_path": image_path,
                "detections": detections,
                "num_detections": len(detections),
                "confidence_threshold": confidence_threshold,
                "success": True,
                "error": None,
            }
            logger.info(
                f"Processed {idx + 1}/{len(dataset)}: {image_path or 'image'} - "
                f"{len(detections)} objects detected"
            )

        except Exception as e:
            # Handle errors gracefully
            logger.error(f"Error processing image at index {idx}: {str(e)}")
            result = {
                "image_path": None,
                "detections": [],
                "num_detections": 0,
                "confidence_threshold": confidence_threshold,
                "success": False,
                "error": str(e),
            }

        # Preserve original columns from dataset
        for col in dataset.columns:
            if col not in result:
                result[col] = row[col]

        results.append(result)

    results_df = pd.DataFrame(results)
    logger.info(
        f"Completed dataset processing: {len(results_df)} images, "
        f"{results_df['success'].sum()} successful"
    )

    return results_df


def draw_detections_on_image(
    image: str | Path | Image.Image,
    detections: list[dict[str, Any]] | dict[str, Any],
    output_path: str | Path | None = None,
    box_style: dict[str, Any] | None = None,
    use_cv2: bool = False,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    This node visualizes object detection results by drawing bounding boxes
    and labels on the image.

    Args:
        image: Input image
        detections: Detection results (list of dicts or dict with 'detections' key)
        output_path: Optional path to save the annotated image
        box_style: Dictionary of visual formatting options with keys:
            - box_color (str): Color for bounding boxes (default: "red")
            - text_color (str): Color for text labels (default: "white")
            - box_width (int): Width of bounding box lines (default: 3)
            - font_size (int): Font size for labels (default: 20)
        use_cv2: Whether to use OpenCV for drawing (alternative to PIL)

    Returns:
        Image with bounding boxes drawn

    Example:
        >>> annotated_img = draw_detections_on_image(image, detections, "output.jpg")
    """
    logger.info("Drawing bounding boxes on image")

    style = box_style or {}
    box_color = style.get("box_color", "red")
    text_color = style.get("text_color", "white")
    box_width = style.get("box_width", 3)
    font_size = style.get("font_size", 20)

    # Load image based on type
    if isinstance(image, dict) and "path" in image:
        # HuggingFace Image format
        image = Image.open(image["path"]).convert("RGB")
    elif isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    # Extract detections list if wrapped in dict
    if isinstance(detections, dict) and "detections" in detections:
        detections = detections["detections"]

    # Draw bounding boxes
    if use_cv2:
        img_array = np.array(image)
        img_with_boxes = draw_bounding_boxes_cv2(
            img_array, detections, box_width=box_width
        )
        # Convert back to PIL Image
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        img_with_boxes = Image.fromarray(img_with_boxes)
    else:
        img_with_boxes = draw_bounding_boxes(
            image,
            detections,
            box_color=box_color,
            text_color=text_color,
            box_width=box_width,
            font_size=font_size,
        )

    # Save if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_with_boxes.save(output_path)
        logger.info(f"Saved annotated image to {output_path}")

    return img_with_boxes


def draw_detections_on_dataset(
    dataset: pd.DataFrame,
    detections_df: pd.DataFrame,
    output_dir: str | Path,
    box_style: dict[str, Any] | None = None,
    use_cv2: bool = False,
) -> pd.DataFrame:
    """
    Draw bounding boxes on multiple images from a dataset.

    This node visualizes object detection results for a batch of images,
    creating annotated images for each one.

    Args:
        dataset: DataFrame with 'image' column containing images
        detections_df: DataFrame with detection results (from detect_objects_in_image)
        output_dir: Directory to save annotated images
        box_style: Dictionary of visual formatting options
        use_cv2: Whether to use OpenCV for drawing

    Returns:
        DataFrame with paths to annotated images

    Example:
        >>> annotated_df = draw_detections_on_dataset(dataset, detections_df, "output/")
    """
    logger.info(f"Drawing bounding boxes on {len(dataset)} images")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, (dataset_row, det_row) in enumerate(
        zip(dataset.iterrows(), detections_df.iterrows())
    ):
        _, dataset_row = dataset_row
        _, det_row = det_row

        try:
            # Get image
            image = dataset_row["image"]

            # Get detections for this image
            detections = det_row.get("detections", [])

            # Skip if no successful detection
            if not det_row.get("success", False):
                logger.warning(f"Skipping image {idx} - detection failed")
                results.append(
                    {
                        "image_idx": idx,
                        "annotated_image_path": None,
                        "success": False,
                    }
                )
                continue

            # Generate output filename
            if isinstance(image, dict) and "path" in image:
                image_filename = Path(image["path"]).name
            elif isinstance(image, (str, Path)):
                image_filename = Path(image).name
            else:
                image_filename = f"image_{idx:04d}.jpg"

            output_path = output_dir / f"annotated_{image_filename}"

            # Draw detections
            annotated_img = draw_detections_on_image(
                image=image,
                detections=detections,
                output_path=output_path,
                box_style=box_style,
                use_cv2=use_cv2,
            )

            results.append(
                {
                    "image_idx": idx,
                    "image": annotated_img,  # Include the PIL image for downstream VLM
                    "annotated_image_path": str(output_path),
                    "num_detections": len(detections),
                    "success": True,
                }
            )

            logger.info(
                f"Saved annotated image {idx + 1}/{len(dataset)}: {output_path}"
            )

        except Exception as e:
            logger.error(f"Error drawing detections for image {idx}: {str(e)}")
            results.append(
                {
                    "image_idx": idx,
                    "annotated_image_path": None,
                    "success": False,
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(results)
    logger.info(
        f"Completed batch visualization: {results_df['success'].sum()}/{len(results_df)} successful"
    )

    return results_df


def save_detection_results(
    results: dict[str, Any] | pd.DataFrame, output_path: str | Path
) -> None:
    """
    Save detection results to disk.

    This node saves detection results in a structured format for later use.

    Args:
        results: Detection results (dict or DataFrame)
        output_path: Path to save results

    Example:
        >>> save_detection_results(results, "data/07_model_output/detections.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, pd.DataFrame):
        # Save as parquet for DataFrames
        if output_path.suffix == ".parquet":
            results.to_parquet(output_path, index=False)
        elif output_path.suffix == ".csv":
            results.to_csv(output_path, index=False)
        else:
            # Default to JSON
            results.to_json(output_path, orient="records", indent=2)
    else:
        # Save as JSON for dicts
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved detection results to {output_path}")


def create_detection_summary(
    detections: dict[str, Any] | pd.DataFrame,
) -> dict[str, Any]:
    """
    Create a summary of detection results.

    This node generates statistics and summaries from detection results.

    Args:
        detections: Detection results

    Returns:
        Summary statistics

    Example:
        >>> summary = create_detection_summary(detection_results)
    """
    logger.info("Creating detection summary")

    if isinstance(detections, pd.DataFrame):
        # Summarize batch results
        total_images = len(detections)
        successful = (
            detections["success"].sum()
            if "success" in detections.columns
            else total_images
        )
        total_detections = (
            detections["num_detections"].sum()
            if "num_detections" in detections.columns
            else 0
        )
        avg_detections = total_detections / total_images if total_images > 0 else 0

        # Count detections by class
        class_counts = {}
        for _, row in detections.iterrows():
            if "detections" in row and isinstance(row["detections"], list):
                for det in row["detections"]:
                    label = det.get("label", "unknown")
                    class_counts[label] = class_counts.get(label, 0) + 1

        summary = {
            "total_images": total_images,
            "successful_images": int(successful),
            "total_detections": int(total_detections),
            "average_detections_per_image": round(avg_detections, 2),
            "detections_by_class": class_counts,
        }
    else:
        # Summarize single image results
        dets = detections.get("detections", [])
        class_counts = {}
        for det in dets:
            label = det.get("label", "unknown")
            class_counts[label] = class_counts.get(label, 0) + 1

        summary = {
            "total_detections": len(dets),
            "detections_by_class": class_counts,
        }

    logger.info(f"Summary created: {summary}")
    return summary
