"""
Kedro nodes for the evaluate_metrics pipeline.

This module implements a comprehensive evaluation pipeline that assesses
Vision Language Model (VLM) output descriptions using:
  - BERTScore: token-level semantic similarity between generated and
    reference descriptions via contextual embeddings.
  - CLIPScore: vision-language alignment between images and generated
    descriptions using CLIP embeddings.

The pipeline accepts VLM outputs from the vision_language_model pipeline,
computes both metrics, and produces detailed evaluation reports.
"""

from __future__ import annotations

import io
import json
import logging
import math
import statistics
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from bert_score import score as bert_score_fn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_pil(image: Any) -> Image.Image | None:
    """Convert various image representations to a PIL Image, or return None."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.exists():
            return Image.open(path).convert("RGB")
        logger.warning(f"Image path does not exist: {image}")
        return None
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    logger.warning(f"Unsupported image type: {type(image)}")
    return None


def _confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Return a symmetric confidence interval using the normal approximation."""

    n = len(values)
    if n < 2:
        mean = values[0] if values else float("nan")
        return (mean, mean)

    mean = statistics.mean(values)
    std = statistics.stdev(values)
    # z-value for 95 % CI ≈ 1.96
    z = 1.96 if abs(confidence - 0.95) < 1e-6 else 2.576  # fallback to 99 %
    margin = z * std / math.sqrt(n)
    return (mean - margin, mean + margin)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Node 1 – prepare_evaluation_data
# ---------------------------------------------------------------------------


def prepare_evaluation_data(
    hf_od_vlm_results: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Validate and prepare the VLM results DataFrame for metric evaluation.

    Performs sanity checks (missing columns, empty texts, image availability),
    filters out failed inference rows if requested, and adds a normalised
    ``reference_text`` column that downstream metric nodes consume.

    Args:
        hf_od_vlm_results: DataFrame produced by the vision_language_model
            pipeline (parquet → MemoryDataset).  Must contain at least the
            column specified by ``eval_candidate_column``.
        parameters: Kedro parameter dict with keys:
            - ``eval_candidate_column`` (str, default ``"vlm_response"``):
              column that holds the VLM-generated description.
            - ``eval_reference_column`` (str | None): column with human
              reference descriptions; ``None`` skips BERTScore.
            - ``eval_image_column`` (str, default ``"image"``): column that
              holds the image (PIL / path / bytes).
            - ``eval_filter_failed`` (bool, default ``True``): drop rows
              where VLM inference failed.

    Returns:
        Cleaned DataFrame ready for metric computation.

    Raises:
        ValueError: if the candidate column is absent from the DataFrame.
    """
    logger.info("Preparing evaluation data …")

    candidate_col: str = parameters.get("eval_candidate_column", "vlm_response")
    reference_col: str | None = parameters.get("eval_reference_column", None)
    image_col: str = parameters.get("eval_image_column", "image")
    filter_failed: bool = parameters.get("eval_filter_failed", True)

    df = hf_od_vlm_results.copy()
    total_rows = len(df)
    logger.info(f"Input DataFrame: {total_rows} rows, columns: {list(df.columns)}")

    # ── Sanity checks ───────────────────────────────────────────────────────
    if candidate_col not in df.columns:
        raise ValueError(
            f"Candidate column '{candidate_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if reference_col and reference_col not in df.columns:
        logger.warning(
            f"Reference column '{reference_col}' not found – BERTScore will be skipped."
        )
        reference_col = None

    # ── Filter failed inferences ────────────────────────────────────────────
    if filter_failed and "vlm_success" in df.columns:
        failed_mask = ~df["vlm_success"].astype(bool)
        n_failed = failed_mask.sum()
        if n_failed:
            logger.warning(f"Dropping {n_failed} rows where vlm_success=False")
        df = df[~failed_mask].reset_index(drop=True)

    # ── Validate candidate texts ────────────────────────────────────────────
    empty_candidate = df[candidate_col].isna() | (
        df[candidate_col].astype(str).str.strip() == ""
    )
    n_empty = empty_candidate.sum()
    if n_empty:
        logger.warning(
            f"{n_empty} rows have empty candidate texts; replacing with '<empty>'."
        )
        df.loc[empty_candidate, candidate_col] = "<empty>"

    # ── Validate reference texts ────────────────────────────────────────────
    if reference_col:
        empty_ref = df[reference_col].isna() | (
            df[reference_col].astype(str).str.strip() == ""
        )
        n_empty_ref = empty_ref.sum()
        if n_empty_ref:
            logger.warning(
                f"{n_empty_ref} rows have empty reference texts; replacing with '<empty>'."
            )
            df.loc[empty_ref, reference_col] = "<empty>"
        df["_reference_text"] = df[reference_col].astype(str)
    else:
        df["_reference_text"] = None

    # ── Normalise column names used by downstream nodes ────────────────────
    df["_candidate_text"] = df[candidate_col].astype(str)
    df["_image_col"] = image_col  # just carry the name as metadata

    # ── Image availability check ────────────────────────────────────────────
    if image_col in df.columns:
        n_images = df[image_col].notna().sum()
        logger.info(
            f"Image column '{image_col}': {n_images}/{len(df)} non-null entries"
        )
    else:
        logger.warning(
            f"Image column '{image_col}' not found – CLIPScore will use text-only mode."
        )

    logger.info(
        f"Prepared {len(df)}/{total_rows} rows for evaluation "
        f"(reference available: {reference_col is not None})"
    )
    return df


# ---------------------------------------------------------------------------
# Node 2 – compute_bert_score
# ---------------------------------------------------------------------------


def compute_bert_score(
    eval_df: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Compute token-level semantic similarity via BERTScore.

    Uses contextual BERT embeddings to measure precision, recall and F1
    between each generated description and its reference.  When reference
    texts are absent (``_reference_text`` is None) the function returns an
    empty-score DataFrame with a diagnostic message.

    Args:
        eval_df: Prepared DataFrame from :func:`prepare_evaluation_data`.
        parameters: Kedro parameter dict with keys:
            - ``bert_model_type`` (str, default ``"distilbert-base-uncased"``):
              HuggingFace model used for BERTScore computation.
            - ``bert_lang`` (str, default ``"en"``): language code passed to
              ``bert_score.score``.
            - ``bert_batch_size`` (int, default ``32``): batch size for
              BERTScore computation.
            - ``bert_device`` (str | None): device override (e.g. ``"cpu"``).
            - ``bert_rescale_with_baseline`` (bool, default ``True``): whether
              to apply baseline rescaling.

    Returns:
        DataFrame with columns:
        ``bert_precision``, ``bert_recall``, ``bert_f1``,
        ``bert_skipped`` (bool flag when evaluation was not run).
    """
    logger.info("Computing BERTScore …")

    # Check if references are available
    if (
        "_reference_text" not in eval_df.columns
        or eval_df["_reference_text"].isna().all()
    ):
        logger.warning(
            "No reference texts available – BERTScore computation skipped. "
            "Set 'eval_reference_column' in parameters to enable BERTScore."
        )
        result_df = eval_df.copy()
        result_df["bert_precision"] = float("nan")
        result_df["bert_recall"] = float("nan")
        result_df["bert_f1"] = float("nan")
        result_df["bert_skipped"] = True
        return result_df

    model_type: str = parameters.get("bert_model_type", "distilbert-base-uncased")
    lang: str = parameters.get("bert_lang", "en")
    batch_size: int = int(parameters.get("bert_batch_size", 32))
    device: str | None = parameters.get("bert_device", None)
    rescale: bool = bool(parameters.get("bert_rescale_with_baseline", True))

    candidates = eval_df["_candidate_text"].tolist()
    references = eval_df["_reference_text"].astype(str).tolist()

    logger.info(
        f"BERTScore: {len(candidates)} pairs | model={model_type} | lang={lang} | "
        f"batch_size={batch_size} | rescale={rescale}"
    )

    try:
        score_kwargs: dict[str, Any] = {
            "cands": candidates,
            "refs": references,
            "lang": lang,
            "model_type": model_type,
            "batch_size": batch_size,
            "rescale_with_baseline": rescale,
            "verbose": False,
        }
        if device is not None:
            score_kwargs["device"] = device

        P, R, F1 = bert_score_fn(**score_kwargs)

        result_df = eval_df.copy()
        result_df["bert_precision"] = P.numpy().tolist()
        result_df["bert_recall"] = R.numpy().tolist()
        result_df["bert_f1"] = F1.numpy().tolist()
        result_df["bert_skipped"] = False

        avg_f1 = float(F1.mean())
        logger.info(
            f"BERTScore completed – avg F1: {avg_f1:.4f} | "
            f"avg P: {float(P.mean()):.4f} | avg R: {float(R.mean()):.4f}"
        )

    except ImportError:
        logger.error(
            "bert_score package not installed. Install it with: pip install bert-score"
        )
        result_df = eval_df.copy()
        result_df["bert_precision"] = float("nan")
        result_df["bert_recall"] = float("nan")
        result_df["bert_f1"] = float("nan")
        result_df["bert_skipped"] = True

    except Exception:
        logger.error(f"BERTScore computation failed:\n{traceback.format_exc()}")
        result_df = eval_df.copy()
        result_df["bert_precision"] = float("nan")
        result_df["bert_recall"] = float("nan")
        result_df["bert_f1"] = float("nan")
        result_df["bert_skipped"] = True

    return result_df


# ---------------------------------------------------------------------------
# Node 3 – compute_clip_score
# ---------------------------------------------------------------------------


def compute_clip_score(
    eval_df: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """Measure vision-language alignment using CLIP embeddings (CLIPScore).

    For each sample computes:
    * ``clip_score``: cosine similarity between the CLIP image embedding and
      the CLIP text embedding of the generated description, scaled to [0, 100].
    * ``ref_clip_score`` (when reference texts are available): same metric
      computed against the reference description.
    * ``ref_clip_score_harmonic``: harmonic mean of ``clip_score`` and
      ``ref_clip_score`` (RefCLIPScore).

    Args:
        eval_df: Prepared DataFrame (output of :func:`prepare_evaluation_data`
            or :func:`compute_bert_score`).
        parameters: Kedro parameter dict with keys:
            - ``clip_model_name`` (str, default ``"openai/clip-vit-base-patch32"``):
              CLIP model identifier (HuggingFace or open_clip format).
            - ``clip_batch_size`` (int, default ``32``): images processed per batch.
            - ``clip_device`` (str, default ``"cpu"``): computation device.
            - ``clip_image_column`` (str, default ``"image"``): DataFrame column
              holding the image data (PIL / path / bytes).
            - ``clip_w`` (float, default ``2.5``): weight multiplier for CLIPScore
              (original paper uses 2.5).

    Returns:
        DataFrame with added columns:
        ``clip_score``, ``clip_image_embed_norm``, ``clip_text_embed_norm``,
        ``ref_clip_score`` (when references available),
        ``ref_clip_score_harmonic`` (RefCLIPScore),
        ``clip_skipped`` (bool).
    """
    logger.info("Computing CLIPScore …")

    model_name: str = parameters.get("clip_model_name", "openai/clip-vit-base-patch32")
    batch_size: int = int(parameters.get("clip_batch_size", 32))
    device: str = parameters.get("clip_device", "cpu")
    image_col: str = parameters.get("clip_image_column", "image")
    w: float = float(parameters.get("clip_w", 2.5))

    result_df = eval_df.copy()

    # ── Resolve image column ────────────────────────────────────────────────
    # eval_df carries _image_col as a scalar string in every row
    if "_image_col" in eval_df.columns:
        detected_col = eval_df["_image_col"].iloc[0] if len(eval_df) else image_col
        image_col = detected_col if detected_col in eval_df.columns else image_col

    if image_col not in eval_df.columns:
        logger.warning(f"Image column '{image_col}' not found – CLIPScore skipped.")
        result_df["clip_score"] = float("nan")
        result_df["clip_skipped"] = True
        return result_df

    try:
        logger.info(f"Loading CLIP model: {model_name} on {device}")
        processor = CLIPProcessor.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name).to(device)
        clip_model.eval()

        candidates = eval_df["_candidate_text"].tolist()
        has_references = (
            "_reference_text" in eval_df.columns
            and not eval_df["_reference_text"].isna().all()
        )
        references = (
            eval_df["_reference_text"].astype(str).tolist() if has_references else []
        )

        clip_scores: list[float] = []
        ref_clip_scores: list[float] = []
        img_norms: list[float] = []
        txt_norms: list[float] = []
        skipped_flags: list[bool] = []

        n = len(eval_df)
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            logger.debug(f"CLIPScore batch {batch_start}–{batch_end} / {n}")

            batch_images: list[Image.Image] = []
            batch_candidates: list[str] = []
            batch_refs: list[str] = []
            valid_indices: list[int] = []

            for i in range(batch_start, batch_end):
                raw_image = eval_df.iloc[i][image_col]
                pil_img = _to_pil(raw_image)
                if pil_img is None:
                    logger.warning(f"Row {i}: could not load image – row skipped.")
                    clip_scores.append(float("nan"))
                    img_norms.append(float("nan"))
                    txt_norms.append(float("nan"))
                    if has_references:
                        ref_clip_scores.append(float("nan"))
                    skipped_flags.append(True)
                    continue

                batch_images.append(pil_img)
                batch_candidates.append(candidates[i])
                if has_references:
                    batch_refs.append(references[i])
                valid_indices.append(i)

            if not batch_images:
                continue

            with torch.no_grad():
                # Encode images
                img_inputs = processor(
                    images=batch_images, return_tensors="pt", padding=True
                )
                img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
                image_feats = clip_model.get_image_features(**img_inputs)
                image_feats_norm = image_feats / image_feats.norm(dim=-1, keepdim=True)

                # Encode candidate texts
                txt_inputs = processor(
                    text=batch_candidates,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
                text_feats = clip_model.get_text_features(**txt_inputs)
                text_feats_norm = text_feats / text_feats.norm(dim=-1, keepdim=True)

                cos_sim = (image_feats_norm * text_feats_norm).sum(dim=-1)
                batch_clip = (w * cos_sim.clamp(min=0)).cpu().numpy()

                batch_img_norms = image_feats.norm(dim=-1).cpu().numpy()
                batch_txt_norms = text_feats.norm(dim=-1).cpu().numpy()

                # Encode reference texts if available
                if has_references:
                    ref_inputs = processor(
                        text=batch_refs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    ref_inputs = {k: v.to(device) for k, v in ref_inputs.items()}
                    ref_feats = clip_model.get_text_features(**ref_inputs)
                    ref_feats_norm = ref_feats / ref_feats.norm(dim=-1, keepdim=True)
                    ref_cos = (image_feats_norm * ref_feats_norm).sum(dim=-1)
                    batch_ref_clip = (w * ref_cos.clamp(min=0)).cpu().numpy()

            for j in range(len(valid_indices)):
                clip_scores.append(float(batch_clip[j]))
                img_norms.append(float(batch_img_norms[j]))
                txt_norms.append(float(batch_txt_norms[j]))
                if has_references:
                    ref_clip_scores.append(float(batch_ref_clip[j]))
                skipped_flags.append(False)

        result_df["clip_score"] = clip_scores
        result_df["clip_image_embed_norm"] = img_norms
        result_df["clip_text_embed_norm"] = txt_norms
        result_df["clip_skipped"] = skipped_flags

        if has_references:
            result_df["ref_clip_score"] = ref_clip_scores
            # RefCLIPScore = harmonic mean of clip_score and ref_clip_score
            harmonic: list[float] = []
            for cs, rcs in zip(clip_scores, ref_clip_scores):
                if cs > 0 and rcs > 0:
                    harmonic.append(2 * cs * rcs / (cs + rcs))
                else:
                    harmonic.append(float("nan"))
            result_df["ref_clip_score_harmonic"] = harmonic

        valid_scores = [s for s in clip_scores if not np.isnan(s)]
        avg_clip = statistics.mean(valid_scores) if valid_scores else float("nan")
        logger.info(
            f"CLIPScore completed – avg score: {avg_clip:.4f} | "
            f"skipped: {sum(skipped_flags)}/{n}"
        )

    except ImportError as exc:
        logger.error(
            f"Required package not installed: {exc}. "
            "Install transformers (already a dependency) and ensure CLIP weights are accessible."
        )
        result_df["clip_score"] = float("nan")
        result_df["clip_skipped"] = True

    except Exception:
        logger.error(f"CLIPScore computation failed:\n{traceback.format_exc()}")
        result_df["clip_score"] = float("nan")
        result_df["clip_skipped"] = True

    return result_df


# ---------------------------------------------------------------------------
# Node 4 – generate_evaluation_report
# ---------------------------------------------------------------------------


def generate_evaluation_report(
    scored_df: pd.DataFrame,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate per-sample scores into a structured evaluation report.

    Combines BERTScore and CLIPScore per-sample results into a comprehensive
    report with aggregate statistics, confidence intervals, and per-sample
    breakdowns.

    Args:
        scored_df: DataFrame produced after both metric computation nodes.
        parameters: Kedro parameter dict with keys:
            - ``eval_confidence_level`` (float, default ``0.95``): CI level.
            - ``eval_top_k`` (int, default ``10``): number of best/worst
              samples to include in the report.
            - ``eval_report_title`` (str): title embedded in the report.

    Returns:
        Nested dict with keys:
        ``metadata``, ``bert_score``, ``clip_score``, ``per_sample``,
        ``top_k_best``, ``top_k_worst``, ``diagnostics``.
    """
    logger.info("Generating evaluation report …")

    confidence_level: float = float(parameters.get("eval_confidence_level", 0.95))
    top_k: int = int(parameters.get("eval_top_k", 10))
    title: str = parameters.get("eval_report_title", "VLM Evaluation Report")

    n = len(scored_df)

    def _agg_metric(col: str) -> dict[str, Any]:
        """Return aggregate stats for a score column."""
        if col not in scored_df.columns:
            return {"available": False}
        vals = scored_df[col].dropna().tolist()
        if not vals:
            return {"available": False, "reason": "all NaN"}
        ci_low, ci_high = _confidence_interval(vals, confidence_level)
        return {
            "available": True,
            "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            f"ci_{int(confidence_level * 100)}_low": ci_low,
            f"ci_{int(confidence_level * 100)}_high": ci_high,
        }

    # ── BERTScore aggregates ────────────────────────────────────────────────
    bert_skipped = bool(scored_df.get("bert_skipped", pd.Series([True])).all())
    bert_section: dict[str, Any] = {
        "skipped": bert_skipped,
        "precision": _agg_metric("bert_precision"),
        "recall": _agg_metric("bert_recall"),
        "f1": _agg_metric("bert_f1"),
    }

    # ── CLIPScore aggregates ────────────────────────────────────────────────
    clip_skipped = bool(scored_df.get("clip_skipped", pd.Series([True])).all())
    clip_section: dict[str, Any] = {
        "skipped": clip_skipped,
        "clip_score": _agg_metric("clip_score"),
    }
    if "ref_clip_score" in scored_df.columns:
        clip_section["ref_clip_score"] = _agg_metric("ref_clip_score")
        clip_section["ref_clip_score_harmonic"] = _agg_metric("ref_clip_score_harmonic")

    # ── Per-sample records ──────────────────────────────────────────────────
    score_cols = [
        c
        for c in [
            "_candidate_text",
            "_reference_text",
            "bert_precision",
            "bert_recall",
            "bert_f1",
            "clip_score",
            "ref_clip_score",
            "ref_clip_score_harmonic",
        ]
        if c in scored_df.columns
    ]
    per_sample = scored_df[score_cols].copy()
    per_sample = per_sample.rename(
        columns={"_candidate_text": "candidate", "_reference_text": "reference"}
    )
    per_sample_records = per_sample.where(pd.notna(per_sample), None).to_dict(
        orient="records"
    )

    # ── Top-K best / worst by primary metric ───────────────────────────────
    primary_metric = "bert_f1" if "bert_f1" in scored_df.columns else "clip_score"
    top_k_best: list[dict] = []
    top_k_worst: list[dict] = []
    if primary_metric in scored_df.columns:
        valid = scored_df[primary_metric].notna()
        sorted_asc = scored_df[valid].sort_values(primary_metric)
        top_k_worst = (
            sorted_asc.head(top_k)[score_cols]
            .rename(
                columns={"_candidate_text": "candidate", "_reference_text": "reference"}
            )
            .where(pd.notna(sorted_asc.head(top_k)[score_cols]), None)
            .to_dict(orient="records")
        )
        top_k_best = (
            sorted_asc.tail(top_k)[score_cols]
            .rename(
                columns={"_candidate_text": "candidate", "_reference_text": "reference"}
            )
            .where(pd.notna(sorted_asc.tail(top_k)[score_cols]), None)
            .to_dict(orient="records")
        )

    # ── Diagnostics ─────────────────────────────────────────────────────────
    diagnostics: dict[str, Any] = {
        "total_samples": n,
        "bert_skipped_count": int(
            scored_df.get("bert_skipped", pd.Series([False])).sum()
        ),
        "clip_skipped_count": int(
            scored_df.get("clip_skipped", pd.Series([False])).sum()
        ),
        "empty_candidates": int(
            (scored_df.get("_candidate_text", pd.Series([])) == "<empty>").sum()
        ),
        "empty_references": int(
            (scored_df.get("_reference_text", pd.Series([])) == "<empty>").sum()
        )
        if "_reference_text" in scored_df.columns
        else None,
    }

    report: dict[str, Any] = {
        "metadata": {
            "title": title,
            "n_samples": n,
            "confidence_level": confidence_level,
            "top_k": top_k,
        },
        "bert_score": bert_section,
        "clip_score": clip_section,
        "per_sample": per_sample_records,
        "top_k_best": top_k_best,
        "top_k_worst": top_k_worst,
        "diagnostics": diagnostics,
    }

    logger.info(
        f"Evaluation report generated: {n} samples | "
        f"BERTScore skipped={bert_skipped} | CLIPScore skipped={clip_skipped}"
    )
    return report


# ---------------------------------------------------------------------------
# Node 5 – save_evaluation_report
# ---------------------------------------------------------------------------


def save_evaluation_report(
    evaluation_report: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Persist evaluation results in JSON, CSV, and HTML formats.

    Writes three output files:
    * ``<output_dir>/evaluation_report.json`` – full report with all metrics.
    * ``<output_dir>/evaluation_scores.csv`` – per-sample scores table.
    * ``<output_dir>/evaluation_report.html`` – self-contained HTML report.

    Args:
        evaluation_report: Dict produced by :func:`generate_evaluation_report`.
        parameters: Kedro parameter dict with keys:
            - ``eval_output_dir`` (str, default ``"data/08_reporting"``):
              directory for output files.
            - ``eval_save_html`` (bool, default ``True``): whether to
              generate the HTML report.
            - ``eval_save_csv`` (bool, default ``True``): whether to save
              the per-sample CSV.

    Returns:
        Summary dict with output file paths and high-level metric averages
        (used as the pipeline's final output / ``eval_summary`` catalog entry).
    """
    logger.info("Saving evaluation report …")

    output_dir = Path(parameters.get("eval_output_dir", "data/08_reporting"))
    save_html: bool = bool(parameters.get("eval_save_html", True))
    save_csv: bool = bool(parameters.get("eval_save_csv", True))

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []

    # ── JSON ────────────────────────────────────────────────────────────────
    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2, default=str)
    saved_files.append(str(json_path))
    logger.info(f"Saved JSON report → {json_path}")

    # ── CSV ─────────────────────────────────────────────────────────────────
    csv_path: Path | None = None
    if save_csv and evaluation_report.get("per_sample"):
        csv_path = output_dir / "evaluation_scores.csv"
        scores_df = pd.DataFrame(evaluation_report["per_sample"])
        scores_df.to_csv(csv_path, index=False)
        saved_files.append(str(csv_path))
        logger.info(f"Saved CSV scores → {csv_path}")

    # ── HTML ────────────────────────────────────────────────────────────────
    html_path: Path | None = None
    if save_html:
        html_path = output_dir / "evaluation_report.html"
        html_content = _render_html_report(evaluation_report)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        saved_files.append(str(html_path))
        logger.info(f"Saved HTML report → {html_path}")

    # ── Build summary ───────────────────────────────────────────────────────
    bert_f1_mean = (
        evaluation_report.get("bert_score", {}).get("f1", {}).get("mean", None)
    )
    clip_score_mean = (
        evaluation_report.get("clip_score", {}).get("clip_score", {}).get("mean", None)
    )

    summary: dict[str, Any] = {
        "n_samples": evaluation_report["metadata"]["n_samples"],
        "bert_f1_mean": bert_f1_mean,
        "clip_score_mean": clip_score_mean,
        "saved_files": saved_files,
        "output_dir": str(output_dir),
    }

    logger.info(
        f"Evaluation complete – BERTScore F1 mean: {bert_f1_mean} | "
        f"CLIPScore mean: {clip_score_mean}"
    )
    return summary


# ---------------------------------------------------------------------------
# HTML report renderer (internal helper)
# ---------------------------------------------------------------------------


def _render_html_report(report: dict[str, Any]) -> str:
    """Render a self-contained HTML evaluation report."""
    meta = report.get("metadata", {})
    title = meta.get("title", "VLM Evaluation Report")
    n = meta.get("n_samples", 0)
    diag = report.get("diagnostics", {})
    bert = report.get("bert_score", {})
    clip = report.get("clip_score", {})

    def _fmt(val: Any, decimals: int = 4) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        try:
            return f"{float(val):.{decimals}f}"
        except (TypeError, ValueError):
            return str(val)

    def _section_table(section: dict, label: str) -> str:
        if section.get("skipped") or not section.get("available"):
            return f"<p><em>{label} not available.</em></p>"
        rows = ""
        for k, v in section.items():
            if k in ("available", "n"):
                continue
            rows += f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>"
        return (
            f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    bert_f1_html = _section_table(bert.get("f1", {}), "BERTScore F1")
    clip_html = _section_table(clip.get("clip_score", {}), "CLIPScore")

    # Per-sample table (limit to 200 rows for readability)
    per_sample = report.get("per_sample", [])[:200]
    if per_sample:
        headers = list(per_sample[0].keys())
        header_row = "".join(f"<th>{h}</th>" for h in headers)
        data_rows = ""
        for row in per_sample:
            cells = "".join(f"<td>{_fmt(row.get(h), 4)}</td>" for h in headers)
            data_rows += f"<tr>{cells}</tr>"
        per_sample_html = (
            f"<table><thead><tr>{header_row}</tr></thead>"
            f"<tbody>{data_rows}</tbody></table>"
        )
    else:
        per_sample_html = "<p><em>No per-sample data.</em></p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2rem; color: #333; }}
  h1, h2 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #2c3e50; color: #fff; }}
  tr:nth-child(even) {{ background: #f2f2f2; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #ecf0f1; border-radius: 8px; padding: 1rem; text-align: center; }}
  .card-value {{ font-size: 2rem; font-weight: bold; color: #2980b9; }}
  .card-label {{ font-size: 0.85rem; color: #7f8c8d; }}
  details summary {{ cursor: pointer; font-weight: bold; margin: 0.5rem 0; }}
</style>
</head>
<body>
<h1>{title}</h1>

<div class="summary-grid">
  <div class="card">
    <div class="card-value">{n}</div>
    <div class="card-label">Total Samples</div>
  </div>
  <div class="card">
    <div class="card-value">{_fmt(bert.get("f1", {}).get("mean"))}</div>
    <div class="card-label">BERTScore F1 (mean)</div>
  </div>
  <div class="card">
    <div class="card-value">{_fmt(clip.get("clip_score", {}).get("mean"))}</div>
    <div class="card-label">CLIPScore (mean)</div>
  </div>
</div>

<h2>BERTScore</h2>
{bert_f1_html}

<h2>CLIPScore</h2>
{clip_html}

<h2>Diagnostics</h2>
<ul>
  <li>Total samples: {diag.get("total_samples", n)}</li>
  <li>BERTScore skipped rows: {diag.get("bert_skipped_count", "N/A")}</li>
  <li>CLIPScore skipped rows: {diag.get("clip_skipped_count", "N/A")}</li>
  <li>Empty candidates: {diag.get("empty_candidates", "N/A")}</li>
  <li>Empty references: {diag.get("empty_references", "N/A")}</li>
</ul>

<details>
  <summary>Per-Sample Scores (first 200 rows)</summary>
  {per_sample_html}
</details>

</body>
</html>"""
