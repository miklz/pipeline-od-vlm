"""
Model-specific implementations for various vision-language models.

This module contains concrete implementations of vision models
that inherit from the base classes.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from pipeline_od_vlm.common.base import BaseVisionModel

logger = logging.getLogger(__name__)


class Qwen3VLModel(BaseVisionModel):
    """
    Qwen3-VL vision-language model implementation.

    Supports the Qwen3-VL family of models for vision-language tasks.
    """

    def load_model(self) -> None:
        """Load Qwen3-VL model and processor."""
        logger.info(f"Loading Qwen3-VL model: {self.config.model_name}")

        model_kwargs = {
            "torch_dtype": self.config.dtype,
            "device_map": self.config.device_map,
        }

        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name, **model_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        logger.info("Qwen3-VL model loaded successfully")

    def prepare_inputs(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for Qwen3-VL model."""
        # Format messages in Qwen3-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to model device
        inputs = inputs.to(self.model.device)

        return inputs

    def generate(self, inputs: dict[str, Any], **kwargs) -> Any:
        """Generate output from Qwen3-VL model."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
        }

        # Add any additional generation parameters
        if self.config.additional_params:
            generation_kwargs.update(self.config.additional_params)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        return generated_ids

    def decode_output(self, output: Any, inputs: dict[str, Any]) -> str | list[str]:
        """Decode Qwen3-VL model output."""
        # Trim the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
        ]

        # Decode to text
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Return single string if batch size is 1
        return output_text[0] if len(output_text) == 1 else output_text


class LlavaNextModel(BaseVisionModel):
    """
    LLaVA-NeXT vision-language model implementation.

    Supports the LLaVA-NeXT family of models (e.g., llava-v1.6-mistral-7b).
    """

    def load_model(self) -> None:
        """Load LLaVA-NeXT model and processor."""
        logger.info(f"Loading LLaVA-NeXT model: {self.config.model_name}")

        model_kwargs = {
            "torch_dtype": self.config.dtype,
            "device_map": self.config.device_map,
        }

        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.config.model_name, **model_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        logger.info("LLaVA-NeXT model loaded successfully")

    def prepare_inputs(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for LLaVA-NeXT model."""
        # Format prompt in LLaVA conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")

        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs

    def generate(self, inputs: dict[str, Any], **kwargs) -> Any:
        """Generate output from LLaVA-NeXT model."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
        }

        if self.config.additional_params:
            generation_kwargs.update(self.config.additional_params)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        return generated_ids

    def decode_output(self, output: Any, inputs: dict[str, Any]) -> str | list[str]:
        """Decode LLaVA-NeXT model output."""
        # Decode the generated tokens
        output_text = self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if len(output_text) == 1 else output_text


class DeepSeekVL2Model(BaseVisionModel):
    """
    DeepSeek-VL2 vision-language model implementation.

    Supports the DeepSeek-VL2 family of models.
    """

    def load_model(self) -> None:
        """Load DeepSeek-VL2 model and processor."""
        logger.info(f"Loading DeepSeek-VL2 model: {self.config.model_name}")

        model_kwargs = {
            "torch_dtype": self.config.dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": True,  # DeepSeek models may require this
        }

        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name, **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        logger.info("DeepSeek-VL2 model loaded successfully")

    def prepare_inputs(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for DeepSeek-VL2 model."""
        # Format conversation
        conversation = [
            {
                "role": "User",
                "content": f"<image>\n{prompt}",
            }
        ]

        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            ),
            return_tensors="pt",
        )

        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs

    def generate(self, inputs: dict[str, Any], **kwargs) -> Any:
        """Generate output from DeepSeek-VL2 model."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
        }

        if self.config.additional_params:
            generation_kwargs.update(self.config.additional_params)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        return generated_ids

    def decode_output(self, output: Any, inputs: dict[str, Any]) -> str | list[str]:
        """Decode DeepSeek-VL2 model output."""
        output_text = self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if len(output_text) == 1 else output_text


class VideoLLaMA3Model(BaseVisionModel):
    """
    VideoLLaMA3 vision-language model implementation.

    Supports the VideoLLaMA3 family of models for image understanding.
    """

    def load_model(self) -> None:
        """Load VideoLLaMA3 model and processor."""
        logger.info(f"Loading VideoLLaMA3 model: {self.config.model_name}")

        model_kwargs = {
            "torch_dtype": self.config.dtype,
            "device_map": self.config.device_map,
            "trust_remote_code": True,
        }

        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_name, **model_kwargs
        )

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        logger.info("VideoLLaMA3 model loaded successfully")

    def prepare_inputs(
        self,
        image: str | Path | Image.Image | list[str | Path | Image.Image],
        prompt: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare inputs for VideoLLaMA3 model."""
        # Format conversation with image and text
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template to get properly formatted prompt with image tokens
        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Process inputs with formatted prompt
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")

        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs

    def generate(self, inputs: dict[str, Any], **kwargs) -> Any:
        """Generate output from VideoLLaMA3 model."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config.max_new_tokens),
        }

        if self.config.additional_params:
            generation_kwargs.update(self.config.additional_params)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        return generated_ids

    def decode_output(self, output: Any, inputs: dict[str, Any]) -> str | list[str]:
        """Decode VideoLLaMA3 model output."""
        output_text = self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0] if len(output_text) == 1 else output_text
