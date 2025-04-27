"""
PEFT-specific utility functions for working with parameter-efficient fine-tuning.

This module provides specialized functions for creating, analyzing, and working
with PEFT models, particularly focusing on LoRA and QLoRA techniques.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType, 
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training
)


def create_lora_config(
    task_type=TaskType.SEQ_CLS,
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=None,
    bias="none"
):
    """
    Create a LoRA configuration for parameter-efficient fine-tuning.
    
    Args:
        task_type: Type of task (sequence classification, causal LM, etc.)
        r: Rank of the update matrices
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of modules to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
        
    Returns:
        LoRA configuration object
    """
    if target_modules is None:
        # Default target modules for attention
        target_modules = ["q_lin", "v_lin", "k_lin", "out_lin"]
        
    return LoraConfig(
        task_type=task_type,
        r=r,                      # Rank of the update matrices
        lora_alpha=lora_alpha,    # Scaling factor
        lora_dropout=lora_dropout, # Dropout probability
        target_modules=target_modules,
        bias=bias,
        inference_mode=False,
    )


def get_lora_model(base_model, lora_config):
    """
    Create a PEFT model with LoRA adapters from a base model.
    
    Args:
        base_model: Base model to adapt
        lora_config: LoRA configuration
        
    Returns:
        Model with LoRA adapters
    """
    # Create the PEFT model
    peft_model = get_peft_model(base_model, lora_config)
    
    # Print parameter information
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    percentage = 100 * trainable_params / total_params
    
    print("\nTrainable parameter analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of parameters being trained: {percentage:.4f}%")
    print(f"Parameter reduction factor: {total_params / trainable_params:.2f}x")
    
    # Use the built-in function to print parameters
    print("\nLoRA model trainable parameters:")
    peft_model.print_trainable_parameters()
    
    return peft_model


def prepare_qlora_model(model_name, num_labels=2, quantization_config=None):
    """
    Create a quantized model for QLoRA (Quantized LoRA).
    
    Args:
        model_name: Name or path of the pre-trained model
        num_labels: Number of classification labels
        quantization_config: Optional quantization configuration
        
    Returns:
        Prepared model ready for QLoRA fine-tuning
    """
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for QLoRA but not installed. "
            "Install it with `pip install bitsandbytes`."
        )
    
    # Default quantization config
    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load the model with quantization
    print("Loading base model with quantization...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Prepare model for quantized training
    print("Preparing model for 8-bit training...")
    qlora_base_model = prepare_model_for_kbit_training(base_model)
    
    return qlora_base_model


def get_qlora_config(r=8, lora_alpha=16, lora_dropout=0.05):
    """
    Create a LoRA configuration suitable for QLoRA.
    
    Args:
        r: Rank of the update matrices (typically lower for QLoRA)
        lora_alpha: Scaling factor
        lora_dropout: Dropout probability
        
    Returns:
        LoRA configuration for QLoRA
    """
    # QLoRA typically uses fewer target modules for efficiency
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,                          # Lower rank for QLoRA
        lora_alpha=lora_alpha,        # Lower alpha
        lora_dropout=lora_dropout,    # Less dropout for stability
        target_modules=["q_lin", "v_lin"],  # Target fewer modules
        bias="none",
        inference_mode=False,
    )


def track_memory_usage(model, device):
    """
    Track memory usage of a model on a specific device.
    
    Args:
        model: The model to analyze
        device: Device where the model is loaded
        
    Returns:
        Dictionary with memory usage information in MB
    """
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        return {
            "allocated_mb": memory_allocated,
            "reserved_mb": memory_reserved
        }
    else:
        # For CPU, use approximate size based on parameters
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
        return {
            "allocated_mb": total_size,
            "reserved_mb": total_size
        }


def get_model_size_estimate(model, as_string=False):
    """
    Get an estimate of model size in MB.
    
    Args:
        model: The model to analyze
        as_string: Whether to return the size as a formatted string
        
    Returns:
        Size of the model in MB (or formatted string if as_string=True)
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB
    
    if as_string:
        if total_size_mb > 1024:
            return f"{total_size_mb/1024:.2f} GB"
        else:
            return f"{total_size_mb:.2f} MB"
    else:
        return total_size_mb


def plot_parameter_efficiency(total_params, trainable_params, output_path=None):
    """
    Create a visualization of parameter efficiency for a PEFT model.
    
    Args:
        total_params: Total number of parameters in the model
        trainable_params: Number of trainable parameters
        output_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Calculate frozen parameters
    frozen_params = total_params - trainable_params
    
    # Create pie chart
    sizes = [trainable_params, frozen_params]
    labels = ['Trainable Parameters', 'Frozen Parameters']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # Explode the trainable parameters
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 14})
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Parameter Efficiency: {trainable_params:,} out of {total_params:,} parameters',
              fontsize=16)
    
    # Add text annotation with exact numbers
    plt.annotate(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)\nFrozen: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)",
                xy=(0.5, 0.05), xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        print(f"Parameter efficiency visualization saved to {output_path}")
    else:
        plt.show()


def predict_sentiment(text, model, tokenizer, return_all=False):
    """
    Predict sentiment for a given text using the best trained model
    
    Args:
        text: Input text string or list of strings
        model: Trained model
        tokenizer: Tokenizer for the model
        return_all: Whether to return all details or just the label
    
    Returns:
        If return_all=True: Dict with prediction, confidence, and label
        If return_all=False: Just the label string
        For lists, returns a list of results
    """
    # Handle both single texts and batches
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    
    # Prepare the inputs
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(next(model.parameters()).device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process outputs
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    
    # Prepare results
    results = []
    for i, pred in enumerate(predictions):
        label = "Positive" if pred == 1 else "Negative"
        confidence = probabilities[i][pred].item()
        
        if return_all:
            results.append({
                "prediction": int(pred),
                "confidence": confidence,
                "label": label,
                "probabilities": {
                    "Negative": probabilities[i][0].item(),
                    "Positive": probabilities[i][1].item()
                }
            })
        else:
            results.append(label)
    
    # Return single result or list based on input
    return results[0] if is_single else results


def load_and_merge_lora_model(base_model_name, adapter_path, output_path=None):
    """
    Load a LoRA adapter and merge it with the base model.
    
    Args:
        base_model_name: Name or path of the base model
        adapter_path: Path to the LoRA adapter weights
        output_path: Path to save the merged model (optional)
        
    Returns:
        The merged model
    """
    from transformers import AutoModelForSequenceClassification
    
    # Load the base model
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,  # Assuming binary classification
    )
    
    # Load LoRA model
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge weights
    merged_model = lora_model.merge_and_unload()
    
    # Save the merged model if output path provided
    if output_path:
        merged_model.save_pretrained(output_path)
        print(f"Merged model saved to {output_path}")
    
    return merged_model
