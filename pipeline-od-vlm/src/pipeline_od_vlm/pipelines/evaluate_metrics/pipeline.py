"""
Evaluate Metrics Pipeline.

This pipeline assesses VLM-generated descriptions using BERTScore
(semantic similarity) and CLIPScore (vision-language alignment).

Data flow
---------
hf_od_vlm_results (parquet)
    └─ prepare_evaluation_data  ──► eval_input_df (memory)
           └─ compute_bert_score ──► bert_scored_df (memory)
                  └─ compute_clip_score ──► clip_scored_df (memory)
                         └─ generate_evaluation_report ──► evaluation_report (memory)
                                └─ save_evaluation_report ──► eval_summary (JSON)
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_bert_score,
    compute_clip_score,
    generate_evaluation_report,
    prepare_evaluation_data,
    save_evaluation_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the evaluate_metrics Kedro pipeline.

    Returns:
        Kedro Pipeline with five sequential nodes:
        prepare → bert_score → clip_score → report → save.
    """
    return pipeline(
        [
            # ── Node 1: Validate and prepare input data ──────────────────
            node(
                func=prepare_evaluation_data,
                inputs={
                    "hf_od_vlm_results": "hf_od_vlm_results",
                    "parameters": "params:evaluate_metrics",
                },
                outputs="eval_input_df",
                name="prepare_evaluation_data",
                tags=["evaluate_metrics", "data_prep"],
            ),
            # ── Node 2: BERTScore semantic similarity ─────────────────────
            node(
                func=compute_bert_score,
                inputs={
                    "eval_df": "eval_input_df",
                    "parameters": "params:evaluate_metrics",
                },
                outputs="bert_scored_df",
                name="compute_bert_score",
                tags=["evaluate_metrics", "bert_score"],
            ),
            # ── Node 3: CLIPScore vision-language alignment ───────────────
            node(
                func=compute_clip_score,
                inputs={
                    "eval_df": "bert_scored_df",
                    "parameters": "params:evaluate_metrics",
                },
                outputs="clip_scored_df",
                name="compute_clip_score",
                tags=["evaluate_metrics", "clip_score"],
            ),
            # ── Node 4: Aggregate metrics into a report dict ──────────────
            node(
                func=generate_evaluation_report,
                inputs={
                    "scored_df": "clip_scored_df",
                    "parameters": "params:evaluate_metrics",
                },
                outputs="evaluation_report",
                name="generate_evaluation_report",
                tags=["evaluate_metrics", "reporting"],
            ),
            # ── Node 5: Save JSON / CSV / HTML reports ─────────────────────
            node(
                func=save_evaluation_report,
                inputs={
                    "evaluation_report": "evaluation_report",
                    "parameters": "params:evaluate_metrics",
                },
                outputs="eval_summary",
                name="save_evaluation_report",
                tags=["evaluate_metrics", "output"],
            ),
        ],
        tags=["evaluate_metrics_pipeline"],
    )
