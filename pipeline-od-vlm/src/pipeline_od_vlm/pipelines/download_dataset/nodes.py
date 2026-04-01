"""
Nodes for the download_dataset pipeline.

This module provides a node to download a HuggingFace dataset and
return it as a MemoryDataset for downstream pipelines.
"""

import logging
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


def download_hf_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: int | None = None,
) -> Any:
    """
    Download a HuggingFace dataset and return it.

    Args:
        dataset_name: HuggingFace dataset identifier
            (e.g., "CERTIFoundation/Maritime-SAR-Optical-Descriptions")
        split: Dataset split to load (e.g., "train", "test")
        max_samples: Maximum number of samples to return (None = all)

    Returns:
        HuggingFace Dataset object (or sliced subset if max_samples is set)
    """

    logger.info(f"Downloading HuggingFace dataset: {dataset_name} (split={split})")

    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Sliced dataset to {len(dataset)} samples")
    else:
        logger.info(f"Loaded full dataset with {len(dataset)} samples")

    return dataset.to_pandas()
