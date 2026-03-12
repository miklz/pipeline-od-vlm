# Search and Rescue at Sea: Object Detection and Vision Language Models Pipeline

## Overview

This project implements an experimental pipeline for evaluating object detection and vision language models in maritime search and rescue operations. The pipeline enables systematic experimentation with various computer vision models to detect persons, vessels, debris, and other objects of interest in marine environments.

## Project Structure

This project is organized using Kedro, a Python framework for creating reproducible, maintainable, and modular data science pipelines. The Kedro structure ensures that experiments are well-organized, versioned, and easily reproducible.

## Prerequisites

- Python 3.8 or higher
- `uv` package manager
- Git

## Installation

### 1. Install uv Package Manager

If you don't have `uv` installed, follow the installation instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

### 2. Clone the Repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 3. Install Dependencies

This project uses `uv` as the Python package manager. Install all dependencies with:

```bash
uv sync
```

### 4. Install Git Hooks

The project includes pre-commit hooks for code quality and consistency. Install them using:

```bash
uvx pre-commit install
```

This will set up automatic checks that run before each commit, ensuring code formatting, linting, and other quality standards are maintained.

## Project Components

### Object Detection Models

The pipeline supports experimentation with various object detection architectures optimized for maritime environments, including detection of:
- Persons in water
- Life rafts and flotation devices
- Vessels and watercraft
- Debris and wreckage
- Marine wildlife

### Vision Language Models

Integration of vision language models enables:
- Scene understanding and description
- Object identification with natural language queries
- Contextual analysis of search and rescue scenarios
- Multi-modal reasoning for decision support

## Running the Pipeline

### Execute the Full Pipeline

```bash
uv run kedro run
```

### Run Specific Pipelines

```bash
uv run kedro run --pipeline <pipeline-name>
```

### Run with Parameters

```bash
uv run kedro run --params <param-key>:<param-value>
```

## Experiment Tracking

All experiments are tracked and versioned using Kedro's built-in experiment tracking capabilities. Results, metrics, and model artifacts are automatically logged and can be compared across different runs.

## Configuration

Configuration files are located in the `conf/` directory:
- `conf/base/`: Base configuration shared across environments
- `conf/local/`: Local configuration (not tracked in git)

Modify parameters in `conf/base/parameters.yml` to adjust model settings, data paths, and experiment configurations.

## Development

### Code Quality

The project enforces code quality through pre-commit hooks that check:
- Code formatting with Ruff formatter
- Linting with Ruff (with automatic fixes)
- Trailing whitespace removal
- End-of-file fixing
- YAML syntax validation
- Large file detection

### Adding New Models

To add new object detection or vision language models, create a new node in the appropriate pipeline and register it in the pipeline definition.

## Data

Place your maritime search and rescue datasets in the `data/` directory following the Kedro data catalog structure:
- `data/01_raw/`: Raw, immutable data
- `data/02_intermediate/`: Intermediate processed data
- `data/03_primary/`: Primary datasets for modeling
- `data/04_feature/`: Feature engineering outputs
- `data/05_model_input/`: Model input datasets
- `data/06_models/`: Trained models
- `data/07_model_output/`: Model predictions and outputs
- `data/08_reporting/`: Reporting outputs

## Contributing

1. Create a new branch for your feature or experiment
2. Make your changes
3. Ensure all pre-commit hooks pass
4. Submit a pull request
