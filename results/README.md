# Results

This directory contains analysis results and performance metrics from model evaluations.

## Directory Structure

- `figures/`: Visualizations generated during model training and evaluation
- CSV and text files with detailed performance metrics and comparison data

## Key Result Files

When running the notebook, the following result files are generated:

- `model_comparison.csv`: Comprehensive metrics comparing all models
- `lora_variants_comparison.csv`: Comparison of different LoRA configurations
- `base_model_classification_report.txt`: Detailed classification metrics for the base model
- `lora_model_classification_report.txt`: Detailed classification metrics for the LoRA model
- `sample_predictions.csv`: Example predictions on test cases with confidence scores

## Figures

The `figures/` subdirectory contains visualizations including:

- `accuracy_comparison.png`: Bar chart comparing accuracy across models
- `parameter_efficiency.png`: Visualization of parameter efficiency
- `confusion_matrices.png`: Confusion matrices for model predictions
- `lora_variants_comparison.png`: Comparison of different LoRA configurations
- `memory_usage_comparison.png`: Memory usage across different models
- `lora_learning_curves.png`: Training and validation curves
- `qlora_learning_curves.png`: Learning curves for QLoRA (if implemented)
- `sample_predictions_*.png`: Visualizations of model predictions

These visualizations are generated automatically when running the notebook.
