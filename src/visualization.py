"""
Visualization utilities for PEFT projects.

This module provides functions for creating visualizations to analyze and compare
the performance of PEFT models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_metric_comparison(metrics_dict, metric_name='accuracy', title=None, output_path=None):
    """
    Create a bar chart comparing a specific metric across different models.
    
    Args:
        metrics_dict: Dictionary mapping model names to metric values
        metric_name: The name of the metric to visualize
        title: Plot title (default derived from metric name)
        output_path: Path to save the visualization (optional)
    """
    plt.figure(figsize=(12, 6))
    models = list(metrics_dict.keys())
    values = [metrics_dict[model][f'eval_{metric_name}'] for model in models]
    
    # Create bar chart with custom colors
    colors = sns.color_palette("muted", len(models))
    bars = plt.bar(models, values, color=colors)
    
    # Set title and labels
    if title is None:
        title = f'{metric_name.capitalize()} Comparison Across Models'
    plt.title(title, fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric_name.capitalize(), fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=11)
    
    # Add a horizontal line for base model reference
    if 'Base Model' in metrics_dict:
        base_value = metrics_dict['Base Model'][f'eval_{metric_name}']
        plt.axhline(y=base_value, color='red', linestyle='--', alpha=0.7, 
                    label=f'Base Model: {base_value:.4f}')
        plt.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Saved {metric_name} comparison chart to {output_path}")
    else:
        plt.show()


def plot_training_history(history, output_path=None):
    """
    Plot training loss and evaluation metrics from training history.
    
    Args:
        history: Training history from Trainer.state.log_history
        output_path: Path to save the plot
    """
    # Extract training loss
    train_loss = []
    train_steps = []
    eval_loss = []
    eval_acc = []
    eval_steps = []
    
    for entry in history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_loss.append(entry['loss'])
            train_steps.append(entry['step'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_acc.append(entry['eval_accuracy'])
            eval_steps.append(entry['step'])
    
    # Create the plot
    plt.figure(figsize=(14, 6))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, 'b-', marker='o', markersize=4, alpha=0.7)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot evaluation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_acc, 'g-', marker='o', markersize=6, label='Accuracy')
    
    # If eval loss exists, plot it on secondary axis
    if eval_loss:
        ax2 = plt.gca().twinx()
        ax2.plot(eval_steps, eval_loss, 'r--', marker='x', markersize=4, alpha=0.7, label='Loss')
        ax2.set_ylabel('Loss', color='r', fontsize=12)
        ax2.tick_params(axis='y', colors='r')
    
    plt.title('Validation Metrics', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Training history plot saved to {output_path}")
    else:
        plt.show()


def visualize_confusion_matrix(predictions, true_labels, output_path=None, model_name="Model"):
    """
    Create and visualize a confusion matrix for the given predictions.
    
    Args:
        predictions: Model predictions (class indices)
        true_labels: True class labels
        output_path: Path to save the visualization (optional)
        model_name: Name of the model for the title
        
    Returns:
        The confusion matrix
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Visualize
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # If labels are binary, add class names
    if cm.shape[0] == 2:
        plt.xticks([0.5, 1.5], ['Negative', 'Positive'])
        plt.yticks([0.5, 1.5], ['Negative', 'Positive'])
    
    plt.tight_layout()
    
    # Save if output path is specified
    if output_path:
        plt.savefig(output_path)
        plt.close()
    
    return cm


def plot_roc_curve(model, dataset, tokenizer, device, output_path=None, model_name="Model"):
    """
    Plot ROC curve for the model on a given dataset.
    
    Args:
        model: The model to evaluate
        dataset: Evaluation dataset
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        output_path: Path to save the visualization
        model_name: Name of the model for the title
    """
    # Create a dataloader for the dataset
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=16)
    
    all_labels = []
    all_probs = []
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].numpy()
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get probabilities for positive class
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_labels.extend(labels)
            all_probs.extend(probs)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"ROC curve saved to {output_path}")
    
    return roc_auc


def create_comprehensive_comparison(all_results, output_path=None):
    """
    Create a comprehensive visual comparison of all models.
    
    Args:
        all_results: Dictionary containing results for all models
        output_path: Path to save the visualization
    """
    # Determine which metrics are available
    metrics = []
    for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
        if f'eval_{metric_name}' in all_results["Base Model"]:
            metrics.append(metric_name)
    
    if not metrics:
        return
        
    # Extract data
    models = list(all_results.keys())
    data = []
    
    for model in models:
        row = [model]
        for metric in metrics:
            if f'eval_{metric}' in all_results[model]:
                row.append(all_results[model][f'eval_{metric}'])
            else:
                row.append(None)
        data.append(row)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data, columns=['Model'] + [m.capitalize() for m in metrics])
    
    # Create a visually appealing table
    fig, ax = plt.subplots(figsize=(12, len(models) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate improvements relative to base model
    improvements = {}
    base_idx = df.index[df['Model'] == 'Base Model'].tolist()[0]
    base_values = df.iloc[base_idx, 1:].values
    
    for i, row in df.iterrows():
        if row['Model'] != 'Base Model':
            improvements[i] = []
            for j, val in enumerate(row.values[1:]):
                if val is not None and base_values[j] is not None:
                    pct = (val - base_values[j]) / base_values[j] * 100
                    improvements[i].append(f"{val:.4f}\n({pct:+.1f}%)")
                else:
                    improvements[i].append("")
    
    # Create the table
    table_data = []
    for i, row in df.iterrows():
        if row['Model'] == 'Base Model':
            table_data.append([row['Model']] + [f"{v:.4f}" for v in row.values[1:]])
        else:
            table_data.append([row['Model']] + improvements[i])
    
    table = ax.table(cellText=table_data, colLabels=df.columns, loc='center',
                     cellLoc='center', colColours=['#f2f2f2']*len(df.columns))
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Adjust row height
    
    # Highlight the best value in each column
    for j in range(1, len(df.columns)):
        col_values = df.iloc[:, j].values
        best_idx = np.nanargmax(col_values)
        if best_idx != base_idx:  # If best is not base model
            cell = table[best_idx+1, j]
            cell.set_facecolor('#d4f7d4')  # Light green
    
    plt.title('Comprehensive Model Comparison', fontsize=16, pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive comparison to {output_path}")
    else:
        plt.show()


def visualize_parameter_comparison(models_info, output_path=None):
    """
    Visualize parameter efficiency for multiple models.
    
    Args:
        models_info: Dictionary of model information including parameters
        output_path: Path to save the visualization
    """
    # Prepare data
    models = []
    total_params = []
    trainable_params = []
    trainable_pct = []
    
    for model_name, info in models_info.items():
        if model_name == "Base Model":
            continue
        if "trainable_params" in info and "total_params" in info:
            models.append(model_name)
            total_params.append(info["total_params"])
            trainable_params.append(info["trainable_params"])
            trainable_pct.append(info["trainable_pct"])
    
    if not models:
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Trainable Parameters Comparison
    y_pos = np.arange(len(models))
    colors = sns.color_palette("viridis", len(models))
    
    bars = ax1.barh(y_pos, trainable_params, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models)
    ax1.invert_yaxis()  # Labels read top-to-bottom
    ax1.set_xlabel('Trainable Parameters')
    ax1.set_title('Trainable Parameters per Model')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                f'{trainable_params[i]:,}', 
                va='center')
    
    # Plot 2: Parameter Efficiency (percentage)
    bars = ax2.barh(y_pos, trainable_pct, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models)
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_xlabel('Percentage of Total Parameters (%)')
    ax2.set_title('Parameter Efficiency')
    
    # Add value annotations
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width * 1.05, bar.get_y() + bar.get_height()/2, 
                f'{trainable_pct[i]:.2f}%', 
                va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Parameter comparison visualization saved to {output_path}")
    else:
        plt.show()


def plot_memory_comparison(memory_data, output_path=None):
    """
    Plot memory usage comparison for different models.
    
    Args:
        memory_data: Dictionary mapping model names to memory usage in MB
        output_path: Path to save the visualization
    """
    if not memory_data or len(memory_data) <= 1:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    models = list(memory_data.keys())
    memory_values = list(memory_data.values())
    
    bars = plt.bar(models, memory_values, color=sns.color_palette("viridis", len(models)))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f} MB',
                ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('Memory Usage Comparison', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add reduction percentages
    if "Base Model" in memory_data:
        base_mem = memory_data["Base Model"]
        for i, (model, mem) in enumerate(zip(models, memory_values)):
            if model != "Base Model":
                reduction = (base_mem - mem) / base_mem * 100
                plt.text(i, mem / 2,
                        f'{reduction:+.1f}%',
                        ha='center', va='center',
                        color='white', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Memory usage comparison saved to {output_path}")
    else:
        plt.show()


def plot_accuracy_vs_efficiency(models_info, output_path=None):
    """
    Create a scatter plot showing accuracy vs parameter efficiency.
    
    Args:
        models_info: Dictionary of model information including accuracy and parameter stats
        output_path: Path to save the visualization
    """
    # Prepare data
    models = []
    accuracies = []
    param_efficiencies = []
    
    for model_name, info in models_info.items():
        if "eval_accuracy" in info and "trainable_pct" in info:
            models.append(model_name)
            accuracies.append(info["eval_accuracy"])
            param_efficiencies.append(info["trainable_pct"])
    
    if len(models) < 2:
        return
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot different colors for base model vs PEFT models
    base_indices = [i for i, model in enumerate(models) if model == "Base Model"]
    peft_indices = [i for i, model in enumerate(models) if model != "Base Model"]
    
    if base_indices:
        plt.scatter([param_efficiencies[i] for i in base_indices], 
                   [accuracies[i] for i in base_indices], 
                   s=100, color='red', label='Base Model')
    
    if peft_indices:
        plt.scatter([param_efficiencies[i] for i in peft_indices], 
                   [accuracies[i] for i in peft_indices], 
                   s=100, c=range(len(peft_indices)), cmap='viridis', label='PEFT Models')
    
    # Add labels for each point
    for i, model in enumerate(models):
        plt.annotate(model, (param_efficiencies[i], accuracies[i]), 
                    fontsize=9, ha='right', va='bottom')
    
    # Add reference line for base model accuracy if available
    base_indices = [i for i, model in enumerate(models) if model == "Base Model"]
    if base_indices:
        base_accuracy = accuracies[base_indices[0]]
        plt.axhline(y=base_accuracy, color='r', linestyle='--', alpha=0.7,
                   label=f'Base Model Accuracy: {base_accuracy:.4f}')
    
    # Labels and title
    plt.xlabel('Percentage of Parameters Trained (%)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Parameter Efficiency', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use log scale for x-axis if there's a wide range of values
    if max(param_efficiencies) / min(param_efficiencies) > 100:
        plt.xscale('log')
        plt.xlabel('Percentage of Parameters Trained (%) - Log Scale', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Accuracy vs efficiency plot saved to {output_path}")
    else:
        plt.show()


def plot_sample_predictions(predictions_data, output_path=None):
    """
    Visualize sentiment predictions for sample texts.
    
    Args:
        predictions_data: List of dictionaries with prediction data
        output_path: Path to save the visualization
    """
    if not predictions_data:
        return
    
    # Create shortened texts for display
    texts_short = [t[:30] + "..." if len(t['text']) > 30 else t['text'] for t in predictions_data]
    
    # Sort by confidence for better visualization
    sorted_data = sorted(predictions_data, key=lambda x: (x['prediction'], x['confidence']), reverse=True)
    
    # Get sorted data
    texts_short = [t[:30] + "..." if len(t['text']) > 30 else t['text'] for t in sorted_data]
    predictions = [d['prediction'] for d in sorted_data]
    confidences = [d['confidence'] for d in sorted_data]
    
    # Create colored confidence bars
    colors = ['green' if p == 1 or p == "Positive" else 'red' for p in predictions]
    
    # Plot bars
    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(texts_short)), confidences, color=colors)
    
    # Add text labels
    for i, bar in enumerate(bars):
        plt.text(i, bar.get_height() + 0.02, texts_short[i],
                rotation=45, ha='right', fontsize=9)
    
    # Customize plot
    plt.ylim(0, 1.1)  # Leave room for text labels
    plt.xticks([])  # Hide x-axis labels as we've added text annotations
    plt.ylabel('Prediction Confidence', fontsize=12)
    plt.title('Sentiment Analysis Prediction Confidence', fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive'),
        Patch(facecolor='red', label='Negative')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Sample predictions visualization saved to {output_path}")
    else:
        plt.show()


def create_radar_chart(models_data, metrics, output_path=None):
    """
    Create a radar chart comparing multiple models across different metrics.
    
    Args:
        models_data: Dictionary mapping model names to metric values
        metrics: List of metrics to include in the chart
        output_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from math import pi
    
    # Number of variables
    N = len(metrics)
    
    # Make sure we have data for all metrics
    valid_metrics = []
    for metric in metrics:
        metric_key = f'eval_{metric}'
        if all(metric_key in models_data[model] for model in models_data):
            valid_metrics.append(metric)
    
    if not valid_metrics:
        return
    
    N = len(valid_metrics)
    
    # Normalize the data for the radar chart
    max_values = {metric: max(models_data[model][f'eval_{metric}'] for model in models_data) 
                 for metric in valid_metrics}
    min_values = {metric: min(models_data[model][f'eval_{metric}'] for model in models_data) 
                 for metric in valid_metrics}
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], [m.capitalize() for m in valid_metrics], size=12)
    
    # Draw y-labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], size=10)
    plt.ylim(0, 1)
    
    # Plot each model
    for i, (model_name, model_metrics) in enumerate(models_data.items()):
        values = []
        for metric in valid_metrics:
            metric_key = f'eval_{metric}'
            # Normalize to 0-1 range
            if max_values[metric] == min_values[metric]:
                norm_value = 1.0
            else:
                value = model_metrics[metric_key]
                norm_value = (value - min_values[metric]) / (max_values[metric] - min_values[metric])
            values.append(norm_value)
        
        # Close the loop
        values += values[:1]
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Comparison', size=15, y=1.1)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Radar chart saved to {output_path}")
    else:
        plt.show()


def plot_learning_curves_comparison(training_histories, output_path=None):
    """
    Compare learning curves from multiple training runs.
    
    Args:
        training_histories: Dictionary mapping model names to training histories
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    for model_name, history in training_histories.items():
        train_loss = []
        train_steps = []
        for entry in history:
            if 'loss' in entry and 'eval_loss' not in entry:
                train_loss.append(entry['loss'])
                train_steps.append(entry['step'])
        if train_loss:
            plt.plot(train_steps, train_loss, marker='o', markersize=2, label=model_name)
    
    plt.title('Training Loss Comparison', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Validation Accuracy
    plt.subplot(2, 2, 2)
    for model_name, history in training_histories.items():
        eval_acc = []
        eval_steps = []
        for entry in history:
            if 'eval_accuracy' in entry:
                eval_acc.append(entry['eval_accuracy'])
                eval_steps.append(entry['step'])
        if eval_acc:
            plt.plot(eval_steps, eval_acc, marker='o', markersize=4, label=model_name)
    
    plt.title('Validation Accuracy Comparison', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Validation Loss
    plt.subplot(2, 2, 3)
    for model_name, history in training_histories.items():
        eval_loss = []
        eval_steps = []
        for entry in history:
            if 'eval_loss' in entry:
                eval_loss.append(entry['eval_loss'])
                eval_steps.append(entry['step'])
        if eval_loss:
            plt.plot(eval_steps, eval_loss, marker='o', markersize=4, label=model_name)
    
    plt.title('Validation Loss Comparison', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Validation F1 Score (if available)
    plt.subplot(2, 2, 4)
    f1_available = False
    for model_name, history in training_histories.items():
        eval_f1 = []
        eval_steps = []
        for entry in history:
            if 'eval_f1' in entry:
                eval_f1.append(entry['eval_f1'])
                eval_steps.append(entry['step'])
                f1_available = True
        if eval_f1:
            plt.plot(eval_steps, eval_f1, marker='o', markersize=4, label=model_name)
    
    if f1_available:
        plt.title('Validation F1 Score Comparison', fontsize=14)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        # If F1 not available, use this space for something else
        plt.title('Training vs Validation Loss', fontsize=14)
        for model_name, history in training_histories.items():
            train_loss_epochs = []
            eval_loss = []
            epochs = []
            
            # Group by epoch
            epoch_data = {}
            for entry in history:
                if 'epoch' in entry:
                    epoch = entry['epoch']
                    if epoch not in epoch_data:
                        epoch_data[epoch] = {'train_loss': [], 'eval_loss': None}
                    
                    if 'loss' in entry and 'eval_loss' not in entry:
                        epoch_data[epoch]['train_loss'].append(entry['loss'])
                    if 'eval_loss' in entry:
                        epoch_data[epoch]['eval_loss'] = entry['eval_loss']
            
            # Calculate average training loss per epoch
            for epoch, data in sorted(epoch_data.items()):
                if data['train_loss'] and data['eval_loss'] is not None:
                    epochs.append(epoch)
                    train_loss_epochs.append(np.mean(data['train_loss']))
                    eval_loss.append(data['eval_loss'])
            
            if epochs:
                plt.plot(epochs, train_loss_epochs, '--', marker='o', label=f'{model_name} Train')
                plt.plot(epochs, eval_loss, '-', marker='x', label=f'{model_name} Val')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Learning curves comparison saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    
    # Simulate some data
    all_results = {
        "Base Model": {
            "eval_accuracy": 0.432,
            "eval_f1": 0.34,
            "eval_precision": 0.26,
            "eval_recall": 0.43
        },
        "LoRA Model": {
            "eval_accuracy": 0.873,
            "eval_f1": 0.87,
            "eval_precision": 0.86,
            "eval_recall": 0.87,
            "trainable_params": 1774084,
            "total_params": 68136964,
            "trainable_pct": 2.60
        },
        "QLoRA Model": {
            "eval_accuracy": 0.845,
            "eval_f1": 0.85,
            "eval_precision": 0.84,
            "eval_recall": 0.84,
            "trainable_params": 770000,
            "total_params": 68136964,
            "trainable_pct": 1.13
        }
    }
    
    # Create example visualizations
    plot_metric_comparison(all_results, 'accuracy', output_path="example_model_comparison.png")
    
    # Example parameter efficiency visualization
    visualize_parameter_comparison({
        "LoRA Model": {"trainable_params": 1774084, "total_params": 68136964, "trainable_pct": 2.60},
        "QLoRA Model": {"trainable_params": 770000, "total_params": 68136964, "trainable_pct": 1.13},
    }, output_path="example_param_efficiency.png")
    
    # Example radar chart
    create_radar_chart(all_results, ['accuracy', 'f1', 'precision', 'recall'], 
                      output_path="example_radar_chart.png")
    
    print("Example visualizations created!")
