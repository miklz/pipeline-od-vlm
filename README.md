# Search and Rescue at Sea: Object Detection and Vision Language Models Pipeline

## Overview

This project implements an experimental Kedro pipeline for evaluating **Object Detection (OD)** and **Vision Language Models (VLM)** in maritime search and rescue (SAR) operations. The pipeline enables systematic experimentation with various computer vision models to detect persons, vessels, debris, and other objects of interest in marine environments, and then uses VLMs to provide natural-language descriptions of those detections.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Object Detection Pipeline](#object-detection-pipeline)
3. [Vision Language Model Pipeline](#vision-language-model-pipeline)
4. [How the Pipelines Are Linked](#how-the-pipelines-are-linked)
5. [Supported Models](#supported-models)
6. [Project Structure](#project-structure)
7. [Data Layer](#data-layer)
8. [Configuration](#configuration)
9. [Installation](#installation)
10. [Running the Pipelines](#running-the-pipelines)
11. [Development](#development)
12. [Contributing](#contributing)

---

## Architecture Overview

The system is composed of four Kedro sub-pipelines that are chained together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Full Pipeline                                │
│                                                                     │
│  ┌───────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────┐  │
│  │ download_     │──▶│ download_    │──▶│  object_   │──▶│ vlm  │  │
│  │ dataset       │   │ model        │   │  detection │   │      │  │
│  └───────────────┘   └──────────────┘   └────────────┘   └──────┘  │
│                                                                     │
│  raw images ────────────────────────────────────────▶ VLM report   │
└─────────────────────────────────────────────────────────────────────┘
```

| Sub-pipeline      | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `download_dataset`| Downloads the maritime SAR dataset from HuggingFace Hub    |
| `download_model`  | Downloads OD model weights (e.g., from MLflow)             |
| `object_detection`| Runs inference, draws bounding boxes, saves results        |
| `vision_language_model` | Runs a VLM on OD-annotated images, saves report     |

---

## Object Detection Pipeline

### Pipeline Graph

```
params:od_model_id ─┐
params:od_parameters─┤
model_path ──────────┴──▶ [load_od_model] ──▶ od_model
                                                   │
maritime_sar_optical_descriptions ─────────────────┤
params:od_confidence_threshold ────────────────────┤
params:od_detection_params ────────────────────────┴──▶ [detect_objects_single]
                                                              │
                                                         od_results
                                                              │
maritime_sar_optical_descriptions ────────────────────────────┤
params:od_batch_output_dir ───────────────────────────────────┤
params:od_box_style ──────────────────────────────────────────┴──▶ [draw_detections_batch]
                                                                          │
                                                                   od_annotated_images ──▶ (to VLM pipeline)
                                                                          │
                                                              [save_annotated_metadata]
                                          od_results ──▶ [save_detection_results]
```

### Nodes

| Node name                | Function                      | Inputs                                                              | Outputs               |
|--------------------------|-------------------------------|---------------------------------------------------------------------|-----------------------|
| `load_od_model`          | `load_object_detection_model` | `od_model_id`, `od_parameters`, `model_path`                        | `od_model`            |
| `detect_objects_single`  | `detect_objects_in_image`     | `od_model`, dataset, `od_confidence_threshold`, `od_detection_params` | `od_results`          |
| `draw_detections_batch`  | `draw_detections_on_dataset`  | dataset, `od_results`, `od_batch_output_dir`, `od_box_style`        | `od_annotated_images` |
| `save_detection_results` | `save_detection_results`      | `od_results`, `od_results_output_path`                              | *(side effect)*       |
| `save_annotated_metadata`| `save_detection_results`      | `od_annotated_images`, `od_annotated_output_path`                   | *(side effect)*       |

### What it does

1. **Model loading** — `load_object_detection_model` uses a factory pattern to instantiate the configured model. If `model_path` is provided (a local path returned by `download_model`), model weights are loaded from disk instead of being fetched from HuggingFace.
2. **Detection** — `detect_objects_in_image` accepts a `pd.DataFrame` (batch) or a single image and returns structured detection records: `label`, `confidence`, `bbox`.
3. **Visualization** — `draw_detections_on_dataset` renders bounding boxes on each image using PIL (or optionally OpenCV) and writes annotated images to `od_batch_output_dir`.
4. **Persistence** — results are saved as JSON / Parquet to `data/07_model_output/`.

### Key parameters (`conf/base/parameters_object_detection.yml`)

```yaml
od_model_id: "rt-deterv2"          # Active model
od_confidence_threshold: 0.5        # Detection confidence cutoff
od_detection_params:
  iou_threshold: 0.45               # NMS IoU threshold (YOLO only)
od_box_style:
  box_color: "red"
  text_color: "white"
  box_width: 3
  font_size: 20
```

---

## Vision Language Model Pipeline

### Pipeline Graph

```
params:vlm_model_id ──┐
params:vlm_parameters ┴──▶ [load_vlm_model] ──▶ vlm_model
                                                     │
od_annotated_images ─────────────────────────────────┤
params:vlm_prompt ───────────────────────────────────┤
params:vlm_image_column ─────────────────────────────┤
params:vlm_inference_params ─────────────────────────┴──▶ [run_vlm_on_annotated_images]
                                                                    │
                                                             vlm_results_df
                                                                    │
params:vlm_output_path ─────────────────────────────────────────────┤
params:vlm_image_column ────────────────────────────────────────────┴──▶ [save_vlm_results]
                                                                               │
                                                                       hf_od_vlm_summary
```

### Nodes

| Node name                    | Function                      | Inputs                                                                              | Outputs             |
|------------------------------|-------------------------------|-------------------------------------------------------------------------------------|---------------------|
| `load_vlm_model`             | `load_vision_model`           | `vlm_model_id`, `vlm_parameters`                                                    | `vlm_model`         |
| `run_vlm_on_annotated_images`| `run_vlm_on_annotated_images` | `vlm_model`, `od_annotated_images`, `vlm_prompt`, `vlm_image_column`, `vlm_inference_params` | `vlm_results_df`    |
| `save_vlm_results`           | `save_hf_od_vlm_results`      | `vlm_results_df`, `vlm_output_path`, `vlm_image_column`                             | `hf_od_vlm_summary` |

### What it does

1. **Model loading** — `load_vision_model` uses the same factory as the OD pipeline to instantiate the chosen VLM (Qwen3-VL, LLaVA-NeXT, DeepSeek-VL2, VideoLLaMA3).
2. **Inference** — `run_vlm_on_annotated_images` iterates over `od_annotated_images` (the DataFrame produced by the OD pipeline). Each row contains a PIL `Image` with bounding boxes already drawn. The VLM receives the annotated image directly — it can *see* the boxes — together with a SAR-domain system prompt.
3. **Saving** — `save_hf_od_vlm_results` serialises the combined OD + VLM result to a Parquet file (PIL images are excluded), and returns a summary dict with counts of successful OD and VLM inferences.

### Default prompt

```text
You are a marine search and rescue (SAR) officer helping.
Your task is to identify the objects detected by our object detection system.
Describe the object in the image inside the bounding box.
```

Modify `vlm_prompt` in `conf/base/parameters_vision_language_model.yml` to change this.

### Key parameters (`conf/base/parameters_vision_language_model.yml`)

```yaml
vlm_model_id: "qwen3-vl"           # Active VLM
vlm_output_path: "data/07_model_output/hf_od_vlm_results.parquet"
vlm_image_column: "image"          # Column holding annotated PIL images
vlm_inference_params: {}           # Extra kwargs forwarded to model.inference()
```

---

## How the Pipelines Are Linked

The key link between the two pipelines is the **`od_annotated_images`** dataset.

```
Object Detection Pipeline                  VLM Pipeline
─────────────────────────────              ─────────────────────────────
draw_detections_batch                      run_vlm_on_annotated_images
        │                                          ▲
        │    od_annotated_images (DataFrame)       │
        └──────────────────────────────────────────┘
             columns: image_idx, image (PIL),
                      annotated_image_path,
                      num_detections, success
```

- The OD pipeline writes a `pd.DataFrame` named `od_annotated_images` into the Kedro in-memory data catalog.
- The VLM pipeline reads `od_annotated_images` as its primary input (`hf_od_annotated_df`).
- Each row carries the **annotated PIL Image** (bounding boxes rendered) alongside OD metadata. The VLM never receives raw detection coordinates — it reasons purely from the visual evidence in the image.

### Pipeline composition in `pipeline_registry.py`

```python
return {
    "__default__": download_pipe + download_model_pipe + od_pipe + vlm_pipe,

    # Run only download + OD
    "object_detection": download_pipe + download_model_pipe + od_pipe,

    # Run the full chain including VLM
    "vision_language_model": download_pipe + download_model_pipe + od_pipe + vlm_pipe,
}
```

The `+` operator merges Kedro pipelines while preserving the data-flow dependencies that Kedro infers from shared dataset names.

---

## Supported Models

### Object Detection

| Model ID     | Class         | Backbone              | Source                      |
|--------------|---------------|-----------------------|-----------------------------|
| `rt-deterv2` | `RTDETRModel` | RT-DETR (ResNet-50vd) | HuggingFace / MLflow        |
| `yolo`       | `YOLOModel`   | YOLOv8–v11            | Ultralytics (`yolov8n.pt`)  |

### Vision Language Models

| Model ID       | Class              | HuggingFace ID                        | Parameters |
|----------------|--------------------|---------------------------------------|------------|
| `qwen3-vl`     | `Qwen3VLModel`     | `Qwen/Qwen3-VL-8B-Instruct`           | ~8B        |
| `llava-next`   | `LlavaNextModel`   | `llava-hf/llava-v1.6-mistral-7b-hf`   | ~7B        |
| `deepseek-vl2` | `DeepSeekVL2Model` | `deepseek-ai/deepseek-vl2`            | varies     |
| `videollama3`  | `VideoLLaMA3Model` | `lkhl/VideoLLaMA3-2B-Image-HF`        | ~2B        |

All VLMs share a common `BaseVisionModel` interface with three abstract methods: `load_model()`, `prepare_inputs()`, `generate()`, and `decode_output()`. New models can be added by implementing this interface and registering them in `common/factory.py`.

---

## Project Structure

```
pipeline-od-vlm/
├── conf/
│   ├── base/
│   │   ├── catalog.yml                         # Dataset definitions
│   │   ├── parameters.yml                      # Global parameters
│   │   ├── parameters_object_detection.yml     # OD pipeline config
│   │   ├── parameters_vision_language_model.yml # VLM pipeline config
│   │   └── parameters_download_model.yml       # Model download config
│   └── local/
│       ├── credentials.yml                     # Secrets (not tracked)
│       └── mlflow.yml                          # MLflow tracking config
├── src/pipeline_od_vlm/
│   ├── pipeline_registry.py                    # Pipeline composition
│   ├── common/
│   │   ├── base.py                             # Abstract base classes
│   │   ├── factory.py                          # Model factory / registry
│   │   └── models.py                           # Shared model utilities
│   └── pipelines/
│       ├── download_dataset/                   # HuggingFace dataset download
│       ├── download_model/                     # MLflow model download
│       ├── object_detection/
│       │   ├── pipeline.py                     # Node wiring
│       │   ├── nodes.py                        # Business logic
│       │   └── models.py                       # RTDETRModel, YOLOModel
│       ├── vision_language_model/
│       │   ├── pipeline.py                     # Node wiring
│       │   ├── nodes.py                        # Business logic
│       │   ├── models.py                       # Qwen3VL, LLaVA, DeepSeek, VideoLLaMA3
│       │   └── utils.py                        # Image loading helpers
│       └── evaluate_metrics/                   # Evaluation nodes
└── data/
    ├── 01_raw/                                 # Immutable raw data
    ├── 06_models/                              # Downloaded model weights
    └── 07_model_output/
        ├── detection_results.json
        ├── annotated_images/                   # Images with bounding boxes
        └── hf_od_vlm_results.parquet           # Final combined output
```

---

## Data Layer

Kedro's data catalog (`conf/base/catalog.yml`) manages all datasets. The most important entries are:

| Catalog name                          | Type              | Description                              |
|---------------------------------------|-------------------|------------------------------------------|
| `maritime_sar_optical_descriptions`   | HuggingFace DS    | Raw maritime SAR image dataset           |
| `model_path`                          | `text.TextDataset`| Local path to downloaded OD model        |
| `od_results`                          | in-memory         | Raw detection records (DataFrame)        |
| `od_annotated_images`                 | in-memory         | Annotated images + OD metadata           |
| `vlm_results_df`                      | in-memory         | VLM responses per image                  |
| `hf_od_vlm_summary`                   | in-memory         | Summary statistics dict                  |

---

## Configuration

All configuration files live under `conf/base/`. Override locally by adding matching files to `conf/local/` (not tracked in git).

### Switch the active OD model

```yaml
# conf/base/parameters_object_detection.yml
od_model_id: "yolo"   # or "rt-deterv2"
```

### Switch the active VLM

```yaml
# conf/base/parameters_vision_language_model.yml
vlm_model_id: "llava-next"   # or "qwen3-vl", "deepseek-vl2", "videollama3"
```

### Adjust detection confidence

```yaml
od_confidence_threshold: 0.6
```

---

## Installation

### 1. Install uv Package Manager

Follow the instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

### 2. Clone the Repository

```bash
git clone <repository-url>
cd pipeline-od-vlm/pipeline-od-vlm
```

### 3. Install Dependencies

```bash
uv sync
```

### 4. Install Git Hooks

```bash
uvx pre-commit install
```

---

## Running the Pipelines

### Full pipeline (download → OD → VLM)

```bash
uv run kedro run
```

### Object Detection only

```bash
uv run kedro run --pipeline object_detection
```

### Full chain including VLM

```bash
uv run kedro run --pipeline vision_language_model
```

### Override a parameter at runtime

```bash
uv run kedro run --pipeline object_detection --params od_model_id:yolo
uv run kedro run --pipeline vision_language_model --params vlm_model_id:llava-next
```

### Run a single node

```bash
uv run kedro run --node load_od_model
```

---

## Development

### Code Quality

Pre-commit hooks enforce:

- **Ruff** — linting with auto-fix
- **Ruff formatter** — code formatting
- Trailing whitespace removal
- End-of-file fixing
- YAML syntax validation
- Large file detection

Run checks manually:

```bash
uvx pre-commit run --all-files
```

### Adding a New Object Detection Model

1. Create a class in `pipelines/object_detection/models.py` that inherits `BaseObjectDetectionModel` and implements `load_model`, `prepare_inputs`, `generate`, `detect_objects`.
2. Register the class in `common/factory.py`.
3. Add model parameters under `od_parameters` in `conf/base/parameters_object_detection.yml`.

### Adding a New Vision Language Model

1. Create a class in `pipelines/vision_language_model/models.py` that inherits `BaseVisionModel` and implements `load_model`, `prepare_inputs`, `generate`, `decode_output`.
2. Register the class in `common/factory.py`.
3. Add model parameters under `vlm_parameters` in `conf/base/parameters_vision_language_model.yml`.

---

## Contributing

1. Create a new branch for your feature or experiment.
2. Make & commit your changes.
3. Submit a pull request.
