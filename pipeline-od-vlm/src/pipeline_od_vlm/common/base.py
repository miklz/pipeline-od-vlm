"""
Base classes and abstractions for vision models.

This module provides the foundation for a plugin-style architecture
that supports multiple vision models including VLMs, object detection,
classification, and segmentation models.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for vision models."""

    model_name: str
    model_folder_name: str
    dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    attn_implementation: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    additional_params: dict[str, Any] | None = None


@dataclass
class InferenceInput:
    """Input data for model inference."""

    image: str | Path | Image.Image | list[str | Path | Image.Image]
    prompt: str
    additional_context: dict[str, Any] | None = None


@dataclass
class InferenceOutput:
    """Output from model inference."""

    text: str | list[str]
    raw_output: Any
    metadata: dict[str, Any] | None = None


class BasePreprocessor(ABC):
    """Base class for preprocessing inputs before model inference."""

    @abstractmethod
    def preprocess(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Preprocess inputs for the model.

        Args:
            image: Input image(s) - can be path, URL, or PIL Image
            prompt: Text prompt for the model
            **kwargs: Additional preprocessing parameters

        Returns:
            Dictionary containing preprocessed inputs ready for the model
        """
        pass


class BasePostprocessor(ABC):
    """Base class for postprocessing model outputs."""

    @abstractmethod
    def postprocess(
        self, raw_output: Any, input_data: dict[str, Any] | None = None, **kwargs
    ) -> InferenceOutput:
        """
        Postprocess model outputs.

        Args:
            raw_output: Raw output from the model
            input_data: Original input data for context
            **kwargs: Additional postprocessing parameters

        Returns:
            Structured inference output
        """
        pass


class BaseVisionModel(ABC):
    """
    Base class for all vision models.

    This abstract class defines the interface that all vision models
    must implement, enabling a plugin-style architecture.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the vision model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.processor = None
        self.preprocessor: BasePreprocessor | None = None
        self.postprocessor: BasePostprocessor | None = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor from pretrained weights."""
        pass

    @abstractmethod
    def prepare_inputs(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare inputs for inference.

        Args:
            image: Input image(s)
            prompt: Text prompt
            **kwargs: Additional parameters

        Returns:
            Prepared inputs for the model
        """
        pass

    @abstractmethod
    def generate(self, inputs: dict[str, Any], **kwargs) -> Any:
        """
        Generate output from the model.

        Args:
            inputs: Prepared model inputs
            **kwargs: Generation parameters

        Returns:
            Raw model output
        """
        pass

    @abstractmethod
    def decode_output(self, output: Any, inputs: dict[str, Any]) -> str | list[str]:
        """
        Decode model output to human-readable format.

        Args:
            output: Raw model output
            inputs: Original inputs for context

        Returns:
            Decoded text output
        """
        pass

    def inference(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> InferenceOutput:
        """
        Run complete inference pipeline.

        Args:
            image: Input image(s)
            prompt: Text prompt
            **kwargs: Additional parameters

        Returns:
            Structured inference output
        """
        logger.info(f"Running inference with {self.__class__.__name__}")

        # Prepare inputs
        inputs = self.prepare_inputs(image, prompt, **kwargs)

        # Generate output
        raw_output = self.generate(inputs, **kwargs)

        # Decode output
        text_output = self.decode_output(raw_output, inputs)

        # Create structured output
        output = InferenceOutput(
            text=text_output,
            raw_output=raw_output,
            metadata={
                "model_name": self.config.model_name,
                "prompt": prompt,
            },
        )

        return output

    def __call__(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> InferenceOutput:
        """Make the model callable for convenience."""
        return self.inference(image, prompt, **kwargs)


class BaseObjectDetectionModel(BaseVisionModel):
    """Base class for object detection models."""

    @abstractmethod
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
        pass


class BaseClassificationModel(BaseVisionModel):
    """Base class for image classification models."""

    @abstractmethod
    def classify(
        self, image: str | Path | Image.Image, top_k: int = 5, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Classify an image.

        Args:
            image: Input image
            top_k: Number of top predictions to return
            **kwargs: Additional parameters

        Returns:
            List of class predictions with probabilities
        """
        pass


class BaseSegmentationModel(BaseVisionModel):
    """Base class for image segmentation models."""

    @abstractmethod
    def segment(self, image: str | Path | Image.Image, **kwargs) -> dict[str, Any]:
        """
        Segment an image.

        Args:
            image: Input image
            **kwargs: Additional parameters

        Returns:
            Segmentation masks and metadata
        """
        pass
