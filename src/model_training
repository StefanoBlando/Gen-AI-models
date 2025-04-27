"""
Model training and evaluation utilities for PEFT projects.

This module provides functions for creating, training, and evaluating PEFT models,
with a focus on LoRA and QLoRA techniques.
"""

import os
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig,
    AutoPeftModelForSequenceClassification,
    prepare_model_for_kbit_training
)
from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute a comprehensive set of evaluation metrics for classification.
    
    Args:
        eval_pred: Tuple of (predictions, labels) from the model
        
    Returns:
        Dictionary of metrics including accuracy, F1, precision, and recall
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Core metrics
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted")
    }
    
    # Per-class metrics if multiple classes are present
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        for label in unique_labels:
            label_predictions = (predictions == label)
            label_true = (labels == label)
            metrics[f"precision_class_{label}"] = precision_score(label_true, label_predictions, zero_division=0)
            metrics[f"recall_class_{label}"] = recall_score(label_true, label_predictions, zero_division=0)
            metrics[f"f1_class_{label}"] = f1_score(label_true, label_predictions, zero_division=0)
    
    return metrics


def create_lora_config(
    task_type: TaskType = TaskType.SEQ_CLS,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None,
    bias: str = "none"
) -> LoraConfig:
    """
    Create a LoRA configuration with specified parameters.
    
    Args:
        task_type: The type of task (sequence classification, causal LM, etc.)
        r: Rank of the update matrices
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of modules to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
        
    Returns:
        LoraConfig: Configuration object for LoRA
    """
    if target_modules is None:
        # Default target modules for sequence classification
        target_modules = ["q_lin", "v_lin", "k_lin", "out_lin"]
    
    return LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        inference_mode=False,
    )


def create_peft_model(
    base_model: PreTrainedModel,
    lora_config: LoraConfig
) -> PeftModel:
    """
    Create a PEFT model from a base model using the specified LoRA configuration.
    
    Args:
        base_model: The pre-trained model to adapt
        lora_config: LoRA configuration
        
    Returns:
        PeftModel: Model with LoRA adapters
    """
    # Create the PEFT model
    peft_model = get_peft_model(base_model, lora_config)
    
    # Print trainable parameters
    print("\nTrainable parameter analysis:")
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    percentage = 100 * trainable_params / total_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of parameters being trained: {percentage:.4f}%")
    print(f"Parameter reduction factor: {total_params / trainable_params:.2f}x")
    
    # Also use the built-in function to print parameters
    print("\nLoRA model trainable parameters:")
    peft_model.print_trainable_parameters()
    
    return peft_model


def create_qlora_model(
    model_name_or_path: str,
    num_labels: int = 2,
    lora_config: Optional[LoraConfig] = None,
    quantization_config: Optional[Dict[str, Any]] = None
) -> PeftModel:
    """
    Create a QLoRA model with quantization and LoRA adapters.
    
    Args:
        model_name_or_path: The name or path of the pre-trained model
        num_labels: Number of classification labels
        lora_config: LoRA configuration (will create a default one if None)
        quantization_config: Quantization settings
        
    Returns:
        PeftModel: Quantized model with LoRA adapters
    """
    # Import bitsandbytes for quantization
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for QLoRA but not installed. "
            "Install it with `pip install bitsandbytes`."
        )
    
    # Default quantization config if not provided
    if quantization_config is None:
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_use_double_quant": True,
        }
    
    # Convert to BitsAndBytesConfig if dictionary
    if isinstance(quantization_config, dict):
        quantization_config = BitsAndBytesConfig(**quantization_config)
    
    # Import the appropriate class based on the model type
    from transformers import AutoModelForSequenceClassification
    
    # Load the base model with quantization
    print("Loading base model with quantization...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        quantization_config=quantization_config,
        device_map="auto",  # Automatically handle device placement
    )
    
    # Prepare model for k-bit training
    print("Preparing model for quantized training...")
    base_model = prepare_model_for_kbit_training(base_model)
    
    # Create default LoRA config if not provided
    if lora_config is None:
        # More conservative settings for QLoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,  # Lower rank
            lora_alpha=16,  # Lower alpha
            lora_dropout=0.05,  # Less dropout for stability
            target_modules=["q_lin", "v_lin"],  # Target fewer modules
            bias="none",
            inference_mode=False,
        )
    
    # Create the PEFT model
    print("Creating QLoRA model...")
    qlora_model = get_peft_model(base_model, lora_config)
    
    # Print parameter information
    print("\nQLoRA model trainable parameters:")
    qlora_model.print_trainable_parameters()
    
    return qlora_model


def create_training_args(
    output_dir: str,
    learning_rate: float = 5e-4,
    batch_size: int = 16,
    eval_batch_size: Optional[int] = None,
    num_epochs: int = 3,
    weight_decay: float = 0.01,
    save_steps: int = 500,
    logging_steps: int = 50,
    use_fp16: bool = True
) -> TrainingArguments:
    """
    Create training arguments for the Hugging Face Trainer.
    
    Args:
        output_dir: Directory to save outputs
        learning_rate: Learning rate
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation (defaults to training batch size)
        num_epochs: Number of training epochs
        weight_decay: Weight decay for regularization
        save_steps: Save checkpoints every X steps
        logging_steps: Log metrics every X steps
        use_fp16: Whether to use mixed precision training
        
    Returns:
        TrainingArguments: Arguments for the Trainer
    """
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=use_fp16 and torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none",
    )


def train_model(
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
    compute_metrics_fn: Callable = compute_metrics
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Train a model using the Hugging Face Trainer.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments
        compute_metrics_fn: Function to compute evaluation metrics
        
    Returns:
        Tuple of (trainer, training_results)
    """
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
    )
    
    # Log training start
    print(f"\nStarting training with {len(train_dataset)} examples")
    print(f"Evaluating on {len(eval_dataset)} examples")
    print(f"Training for {training_args.num_train_epochs} epochs with batch size {training_args.per_device_train_batch_size}")
    
    # Train the model with timing
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    # Print training statistics
    print(f"\nTraining completed in {formatted_time}")
    print(f"Training loss: {train_result.training_loss:.4f}")
    
    # Return trainer and training results
    return trainer, {
        "training_loss": train_result.training_loss,
        "training_time": training_time,
        "formatted_time": formatted_time
    }


def evaluate_model(
    trainer: Trainer,
    eval_dataset: Optional[Dataset] = None
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        trainer: Trained Trainer object
        eval_dataset: Optional evaluation dataset (uses trainer's eval_dataset if None)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Use provided eval dataset or trainer's default
    if eval_dataset is not None:
        original_eval_dataset = trainer.eval_dataset
        trainer.eval_dataset = eval_dataset
    
    # Evaluate model
    print("\nEvaluating the model...")
    eval_results = trainer.evaluate()
    
    # Print evaluation results
    print("\nEvaluation results:")
    print("-" * 50)
    for metric, value in eval_results.items():
        # Skip metrics that are not relevant for display
        if not metric.startswith('eval_runtime') and not metric.startswith('eval_samples_per'):
            print(f"{metric:25s}: {value:.4f}")
    
    # Restore original eval dataset if we changed it
    if eval_dataset is not None and original_eval_dataset is not None:
        trainer.eval_dataset = original_eval_dataset
    
    return eval_results


def save_trained_model(
    model: Union[PreTrainedModel, PeftModel],
    save_path: str
) -> str:
    """
    Save a trained model.
    
    Args:
        model: Trained model to save
        save_path: Directory to save the model
        
    Returns:
        Path where the model was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the model
    model.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")
    
    return save_path


def load_peft_model(
    model_path: str,
    device: Optional[str] = None
) -> PeftModel:
    """
    Load a saved PEFT model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on (e.g., "cuda", "cpu")
        
    Returns:
        Loaded PEFT model
    """
    try:
        # Load the PEFT configuration
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # Determine the task type and load the appropriate model class
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_path)
        
        # Move to specified device if provided
        if device is not None:
            model = model.to(device)
        
        print(f"Model loaded successfully from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        # Try an alternative approach
        print("Trying alternative loading method...")
        
        from transformers import AutoModelForSequenceClassification
        
        # Load the base model
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=2  # Assuming binary classification
        )
        
        # Load the PEFT adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Move to specified device if provided
        if device is not None:
            model = model.to(device)
        
        print(f"Model loaded successfully using alternative method from {model_path}")
        return model


def predict_sentiment(
    text: Union[str, List[str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    return_scores: bool = False,
    max_length: int = 128
) -> Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Predict sentiment for a given text using a trained model.
    
    Args:
        text: Input text string or list of strings
        model: Trained model
        tokenizer: Tokenizer for the model
        return_scores: Whether to return confidence scores
        max_length: Maximum sequence length
        
    Returns:
        If return_scores=False: Prediction string or list of strings
        If return_scores=True: Dictionary or list of dictionaries with predictions and scores
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
        max_length=max_length
    )
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
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
        
        if return_scores:
            results.append({
                "prediction": label,
                "confidence": confidence,
                "probabilities": {
                    "Negative": probabilities[i][0].item(),
                    "Positive": probabilities[i][1].item()
                }
            })
        else:
            results.append(label)
    
    # Return single result or list based on input
    return results[0] if is_single else results


def generate_classification_report(
    model: PreTrainedModel,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer
) -> str:
    """
    Generate a detailed classification report for a model on a dataset.
    
    Args:
        model: Trained model
        dataset: Evaluation dataset
        tokenizer: Tokenizer for the model
        
    Returns:
        String containing the classification report
    """
    # Create a dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16)
    
    all_predictions = []
    all_labels = []
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
    # Generate the report
    report = classification_report(
        all_labels, 
        all_predictions,
        target_names=['Negative', 'Positive'],
        digits=4
    )
    
    return report


def track_memory_usage(model: PreTrainedModel, device: torch.device) -> Dict[str, float]:
    """
    Track memory usage of a model on a specific device.
    
    Args:
        model: The model to analyze
        device: Device where the model is loaded
        
    Returns:
        Dictionary with memory usage statistics in MB
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


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    import torch
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)
    
    # Create LoRA config
    lora_config = create_lora_config()
    
    # Create PEFT model
    peft_model = create_peft_model(base_model, lora_config)
    
    print("PEFT model created successfully")
