"""
Utility functions for vision model pipeline.

This module provides common utility functions for image processing,
data handling, and other shared operations.
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(image_source: str | Path | Image.Image) -> Image.Image:
    """
    Load an image from various sources.

    Args:
        image_source: Can be a file path, URL, or PIL Image

    Returns:
        PIL Image object

    Raises:
        ValueError: If image source is invalid
    """
    if isinstance(image_source, Image.Image):
        return image_source

    if isinstance(image_source, (str, Path)):
        source_str = str(image_source)

        # Check if it's a URL
        if source_str.startswith(("http://", "https://")):
            try:
                response = requests.get(source_str, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                raise ValueError(f"Failed to load image from URL: {e}")

        # Otherwise, treat as file path
        try:
            return Image.open(source_str)
        except Exception as e:
            raise ValueError(f"Failed to load image from path: {e}")

    raise ValueError(f"Unsupported image source type: {type(image_source)}")


def load_images(
    image_sources: str | Path | Image.Image | list[str | Path | Image.Image],
) -> list[Image.Image]:
    """
    Load multiple images from various sources.

    Args:
        image_sources: Single image or list of images

    Returns:
        List of PIL Image objects
    """
    if not isinstance(image_sources, list):
        image_sources = [image_sources]

    return [load_image(source) for source in image_sources]


def save_image(image: Image.Image, path: str | Path) -> None:
    """
    Save a PIL Image to disk.

    Args:
        image: PIL Image to save
        path: Destination path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    logger.info(f"Saved image to {path}")


def format_inference_results(
    results: str | list[str] | dict[str, Any], include_metadata: bool = True
) -> dict[str, Any]:
    """
    Format inference results into a standardized structure.

    Args:
        results: Raw inference results
        include_metadata: Whether to include metadata

    Returns:
        Formatted results dictionary
    """
    if isinstance(results, str):
        return {"text": results, "metadata": {} if include_metadata else None}

    if isinstance(results, list):
        return {
            "text": results,
            "count": len(results),
            "metadata": {} if include_metadata else None,
        }

    if isinstance(results, dict):
        return results

    return {"raw": results}


def batch_images(images: list[Image.Image], batch_size: int) -> list[list[Image.Image]]:
    """
    Batch images for processing.

    Args:
        images: List of images
        batch_size: Size of each batch

    Returns:
        List of image batches
    """
    return [images[i : i + batch_size] for i in range(0, len(images), batch_size)]


def validate_model_config(config: dict[str, Any]) -> bool:
    """
    Validate model configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["model_name"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    return True


def merge_configs(
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def parse_detection_output(
    text_output: str, format_type: str = "list"
) -> list[dict[str, Any]]:
    """
    Parse detection output from text.

    Args:
        text_output: Text output from model
        format_type: Expected format type

    Returns:
        List of detected objects
    """
    # This is a placeholder implementation
    # Actual parsing would depend on the model's output format
    detections = []

    # Simple line-based parsing
    lines = text_output.strip().split("\n")
    for line in lines:
        if line.strip():
            detections.append(
                {
                    "label": line.strip(),
                    "confidence": 1.0,  # Placeholder
                }
            )

    return detections


def calculate_image_stats(image: Image.Image) -> dict[str, Any]:
    """
    Calculate basic statistics for an image.

    Args:
        image: PIL Image

    Returns:
        Dictionary of image statistics
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "size_bytes": len(image.tobytes()) if hasattr(image, "tobytes") else None,
    }
