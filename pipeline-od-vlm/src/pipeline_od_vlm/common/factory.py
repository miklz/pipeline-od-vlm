"""
Model factory and registry for vision models.

This module implements a plugin-style architecture that allows
easy registration and instantiation of different vision models.
"""

import logging
from typing import Any

import torch

from .base import BaseVisionModel, ModelConfig
from .models import (
    DeepSeekVL2Model,
    LlavaNextModel,
    Qwen3VLModel,
    RTDETRModel,
    VideoLLaMA3Model,
    YOLOModel,
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for vision models.

    This class maintains a mapping of model identifiers to their
    implementation classes, enabling a plugin-style architecture.
    """

    _registry: dict[str, type[BaseVisionModel]] = {}

    @classmethod
    def register(cls, model_id: str, model_class: type[BaseVisionModel]) -> None:
        """
        Register a model class with an identifier.

        Args:
            model_id: Unique identifier for the model
            model_class: Model class to register
        """
        if model_id in cls._registry:
            logger.warning(f"Model '{model_id}' is already registered. Overwriting.")

        cls._registry[model_id] = model_class
        logger.info(f"Registered model: {model_id} -> {model_class.__name__}")

    @classmethod
    def get(cls, model_id: str) -> type[BaseVisionModel] | None:
        """
        Get a model class by identifier.

        Args:
            model_id: Model identifier

        Returns:
            Model class if found, None otherwise
        """
        return cls._registry.get(model_id)

    @classmethod
    def list_models(cls) -> list[str]:
        """
        List all registered model identifiers.

        Returns:
            List of registered model IDs
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_id: str) -> bool:
        """
        Check if a model is registered.

        Args:
            model_id: Model identifier

        Returns:
            True if registered, False otherwise
        """
        return model_id in cls._registry


# Register built-in models
# Vision Language Models
ModelRegistry.register("qwen3-vl", Qwen3VLModel)
ModelRegistry.register("llava-next", LlavaNextModel)
ModelRegistry.register("deepseek-vl2", DeepSeekVL2Model)
ModelRegistry.register("videollama3", VideoLLaMA3Model)

# Object Detection Models
ModelRegistry.register("rt-deterv2", RTDETRModel)
ModelRegistry.register("yolo", YOLOModel)


class ModelFactory:
    """
    Factory for creating vision model instances.

    This class provides methods to create model instances from
    configuration dictionaries or ModelConfig objects.
    """

    @staticmethod
    def create_model(
        model_id: str, config: ModelConfig, load_immediately: bool = True
    ) -> BaseVisionModel:
        """
        Create a model instance.

        Args:
            model_id: Model identifier (must be registered)
            config: Model configuration
            load_immediately: Whether to load the model immediately

        Returns:
            Instantiated model

        Raises:
            ValueError: If model_id is not registered
        """
        model_class = ModelRegistry.get(model_id)

        if model_class is None:
            available_models = ModelRegistry.list_models()
            raise ValueError(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available_models}"
            )

        logger.info(f"Creating model instance: {model_id}")
        model = model_class(config)

        if load_immediately:
            model.load_model()

        return model

    @staticmethod
    def create_model_from_dict(
        model_id: str, config_dict: dict[str, Any], load_immediately: bool = True
    ) -> BaseVisionModel:
        """
        Create a model instance from a configuration dictionary.

        Args:
            model_id: Model identifier
            config_dict: Configuration dictionary
            load_immediately: Whether to load the model immediately

        Returns:
            Instantiated model
        """
        # Convert dtype string to torch.dtype if needed
        if "dtype" in config_dict and isinstance(config_dict["dtype"], str):
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            config_dict["dtype"] = dtype_map.get(config_dict["dtype"], torch.bfloat16)

        config = ModelConfig(**config_dict)
        return ModelFactory.create_model(model_id, config, load_immediately)

    @staticmethod
    def create_model_from_params(
        model_id: str, params: dict[str, Any], load_immediately: bool = True
    ) -> BaseVisionModel:
        """
        Create a model from Kedro parameters.

        This method is designed to work with Kedro's parameter system,
        extracting the necessary configuration from the parameters dict.

        Args:
            model_id: Model identifier
            params: Kedro parameters dictionary
            load_immediately: Whether to load the model immediately

        Returns:
            Instantiated model
        """
        # Extract model-specific parameters
        model_params = params.get(model_id, {})

        if not model_params:
            raise ValueError(f"No parameters found for model '{model_id}' in params")

        return ModelFactory.create_model_from_dict(
            model_id, model_params, load_immediately
        )


def get_model_for_name(model_name: str) -> str:
    """
    Map a model name to its model_id.

    This helper function maps Hugging Face model names to their
    corresponding model_id in the registry.

    Args:
        model_name: Hugging Face model name

    Returns:
        Model ID for the registry
    """
    # Define mapping from model names to model IDs
    name_to_id = {
        "Qwen/Qwen3-VL-8B-Instruct": "qwen3-vl",
        "llava-hf/llava-v1.6-mistral-7b-hf": "llava-next",
        "deepseek-ai/deepseek-vl2": "deepseek-vl2",
        "lkhl/VideoLLaMA3-2B-Image-HF": "videollama3",
    }

    # Try exact match first
    if model_name in name_to_id:
        return name_to_id[model_name]

    # Try partial matching
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower and "vl" in model_name_lower:
        return "qwen3-vl"
    elif "llava" in model_name_lower:
        return "llava-next"
    elif "deepseek" in model_name_lower:
        return "deepseek-vl2"
    elif "videollama" in model_name_lower:
        return "videollama3"

    raise ValueError(
        f"Could not determine model_id for model name: {model_name}. "
        f"Please register the model or update the mapping."
    )
