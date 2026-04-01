"""
Object detection model implementations.

This module contains concrete implementations of object detection models
that inherit from BaseObjectDetectionModel.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from pipeline_od_vlm.common.base import (
    BaseObjectDetectionModel,
)

logger = logging.getLogger(__name__)


class RTDETRModel(BaseObjectDetectionModel):
    """
    RT-DETR (Real-Time DEtection TRansformer) model implementation.

    Supports RT-DETR models for real-time object detection with transformers.
    """

    def load_model(self) -> None:
        """Load RT-DETR model and processor.

        ``self.config.model_name`` can be either a HuggingFace model ID
        (e.g. ``"PekingU/rtdetr_r50vd"``) or a local filesystem path to a
        previously downloaded model directory (e.g. returned by the
        ``download_model`` pipeline node).  ``AutoModelForObjectDetection``
        and ``AutoImageProcessor`` both support local paths transparently.
        """
        logger.info(f"Loading RT-DETR model from: {self.config.model_name}")

        model_kwargs = {
            "torch_dtype": self.config.dtype,
            "device_map": self.config.device_map,
        }

        self.model = AutoModelForObjectDetection.from_pretrained(
            self.config.model_name, **model_kwargs
        )

        self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
        logger.info("RT-DETR model loaded successfully")

    def prepare_inputs(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for RT-DETR model."""
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(
            dtype=torch.bfloat16
        )

        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Store original image for later use
        inputs["_original_image"] = image

        return inputs

    def generate(self, inputs: dict[str, Any], **kwargs) -> Any:
        """Generate detections from RT-DETR model."""
        # Extract original image before inference
        original_image = inputs.pop("_original_image", None)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Restore original image to inputs
        if original_image is not None:
            inputs["_original_image"] = original_image

        return outputs

    def decode_output(self, output: Any, inputs: dict[str, Any]) -> str | list[str]:
        """Decode RT-DETR model output to detection results."""
        original_image = inputs.get("_original_image")

        # Get image size
        target_sizes = torch.tensor([original_image.size[::-1]])

        # Post-process outputs
        results = self.processor.post_process_object_detection(
            output, target_sizes=target_sizes, threshold=0.5
        )[0]

        # Format detections as text
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.model.config.id2label[label.item()]
            detections.append(f"{label_name}: {score.item():.3f} at {box}")

        return "\n".join(detections) if detections else "No objects detected"

    def detect_objects(
        self,
        image: str | Path | Image.Image,
        confidence_threshold: float = 0.5,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Detect objects in an image.

        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detections
            **kwargs: Additional parameters

        Returns:
            List of detected objects with bounding boxes and labels
        """
        # Prepare inputs
        inputs = self.prepare_inputs(image, **kwargs)
        original_image = inputs.get("_original_image")

        # Generate detections
        outputs = self.generate(inputs, **kwargs)

        # Get image size
        target_sizes = torch.tensor([original_image.size[::-1]])

        # Post-process outputs
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]

        # Format detections
        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append(
                {
                    "label": self.model.config.id2label[label.item()],
                    "confidence": score.item(),
                    "bbox": box.tolist(),
                }
            )

        return detections


class YOLOModel(BaseObjectDetectionModel):
    """
    YOLO (You Only Look Once) model implementation using Ultralytics.

    Supports YOLOv8, YOLOv9, YOLOv10, and YOLO11 models.
    """

    def load_model(self) -> None:
        """Load YOLO model."""
        logger.info(f"Loading YOLO model: {self.config.model_name}")

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for YOLO models. "
                "Install it with: pip install ultralytics"
            )

        # Load YOLO model
        self.model = YOLO(self.config.model_name)

        # Set device
        if self.config.device_map != "auto":
            device = self.config.device_map
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)
        logger.info(f"YOLO model loaded successfully on {device}")

    def prepare_inputs(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for YOLO model."""
        # YOLO can handle paths, PIL Images, or numpy arrays directly
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        return {
            "image": image,
            "conf": kwargs.get("confidence_threshold", 0.5),
            "iou": kwargs.get("iou_threshold", 0.45),
        }

    def generate(self, inputs: dict[str, Any], **kwargs) -> Any:
        """Generate detections from YOLO model."""
        image = inputs["image"]
        conf = inputs.get("conf", 0.5)
        iou = inputs.get("iou", 0.45)

        # Run inference
        results = self.model(image, conf=conf, iou=iou, verbose=False)

        return results

    def decode_output(self, output: Any, inputs: dict[str, Any]) -> str | list[str]:
        """Decode YOLO model output to detection results."""
        results = output[0]  # Get first result

        # Format detections as text
        detections = []
        for box in results.boxes:
            label = results.names[int(box.cls)]
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()
            bbox = [round(i, 2) for i in bbox]

            detections.append(f"{label}: {confidence:.3f} at {bbox}")

        return "\n".join(detections) if detections else "No objects detected"

    def detect_objects(
        self,
        image: str | Path | Image.Image,
        confidence_threshold: float = 0.5,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Detect objects in an image.

        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detections
            **kwargs: Additional parameters

        Returns:
            List of detected objects with bounding boxes and labels
        """
        # Prepare inputs
        inputs = self.prepare_inputs(
            image, confidence_threshold=confidence_threshold, **kwargs
        )

        # Generate detections
        results = self.generate(inputs, **kwargs)
        result = results[0]  # Get first result

        # Format detections
        detections = []
        for box in result.boxes:
            detections.append(
                {
                    "label": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                }
            )

        return detections


def draw_bounding_boxes(
    image: Image.Image,
    detections: list[dict[str, Any]],
    box_color: str = "red",
    text_color: str = "white",
    box_width: int = 3,
    font_size: int = 20,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: Input PIL Image
        detections: List of detections with bbox, label, and confidence
        box_color: Color for bounding boxes
        text_color: Color for text labels
        box_width: Width of bounding box lines
        font_size: Font size for labels

    Returns:
        Image with bounding boxes drawn
    """
    # Create a copy to avoid modifying the original
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()

    for detection in detections:
        bbox = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]

        # Draw bounding box
        draw.rectangle(bbox, outline=box_color, width=box_width)

        # Prepare label text
        text = f"{label}: {confidence:.2f}"

        # Get text bounding box for background
        text_bbox = draw.textbbox((bbox[0], bbox[1] - font_size - 5), text, font=font)

        # Draw background for text
        draw.rectangle(text_bbox, fill=box_color)

        # Draw text
        draw.text((bbox[0], bbox[1] - font_size - 5), text, fill=text_color, font=font)

    return img_with_boxes


def draw_bounding_boxes_cv2(
    image: np.ndarray | Image.Image,
    detections: list[dict[str, Any]],
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    box_width: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes on an image using OpenCV.

    Args:
        image: Input image (numpy array or PIL Image)
        detections: List of detections with bbox, label, and confidence
        box_color: Color for bounding boxes (BGR format)
        text_color: Color for text labels (BGR format)
        box_width: Width of bounding box lines
        font_scale: Font scale for labels

    Returns:
        Image with bounding boxes drawn (numpy array)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create a copy to avoid modifying the original
    img_with_boxes = image.copy()

    for detection in detections:
        bbox = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]

        # Convert bbox to integers
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), box_color, box_width)

        # Prepare label text
        text = f"{label}: {confidence:.2f}"

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Draw background for text
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            box_color,
            -1,
        )

        # Draw text
        cv2.putText(
            img_with_boxes,
            text,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
        )

    return img_with_boxes
