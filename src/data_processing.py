"""
Data loading and preprocessing utilities for PEFT projects.

This module provides functions for loading, preprocessing, and analyzing datasets
for parameter-efficient fine-tuning projects.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer


def load_sst2_dataset() -> DatasetDict:
    """
    Load the SST-2 dataset with multiple fallback methods in case of errors.
    
    Returns:
        DatasetDict: The loaded dataset or raises an error if all methods fail.
    """
    # Method 1: Standard loading
    try:
        print("Attempting to load SST-2 dataset directly...")
        dataset = load_dataset("glue", "sst2")
        print("Dataset loaded successfully!")
        return dataset
    except Exception as e:
        print(f"Error in standard loading: {e}")
    
    # Method 2: Try with dataset builder
    try:
        print("Attempting to load with dataset builder...")
        from datasets import load_dataset_builder
        builder = load_dataset_builder("glue", "sst2")
        builder.download_and_prepare()
        dataset = builder.as_dataset()
        print("Dataset loaded successfully with builder!")
        return dataset
    except Exception as e:
        print(f"Error with dataset builder: {e}")
    
    # Method 3: Manual download
    try:
        print("Attempting direct download of SST-2...")
        # Download raw data if not already present
        if not os.path.exists('SST-2.zip'):
            os.system("wget -q https://dl.fbaipublicfiles.com/glue/data/SST-2.zip")
            os.system("unzip -q SST-2.zip")
        elif not os.path.exists('SST-2'):
            os.system("unzip -q SST-2.zip")
        
        # Load train and dev data
        train_df = pd.read_csv('SST-2/train.tsv', sep='\t')
        dev_df = pd.read_csv('SST-2/dev.tsv', sep='\t')
        
        # Rename columns to match expected format if needed
        if 'sentence' not in train_df.columns and 'text' in train_df.columns:
            train_df = train_df.rename(columns={'text': 'sentence'})
            dev_df = dev_df.rename(columns={'text': 'sentence'})
        
        # Convert to datasets format
        train_dataset = Dataset.from_pandas(train_df)
        validation_dataset = Dataset.from_pandas(dev_df)
        
        # Create dataset dictionary
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset
        })
        print("Dataset loaded successfully via direct download!")
        return dataset
    except Exception as e:
        print(f"Error with direct download: {e}")
        raise ValueError("All dataset loading methods failed. Please check your environment.")


def create_data_subsets(
    dataset: DatasetDict, 
    train_size: int, 
    validation_size: Optional[int] = None, 
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create smaller subsets of the dataset for training and evaluation.
    
    Args:
        dataset: Original dataset dictionary
        train_size: Number of examples to use for training
        validation_size: Number of examples to use for quick validation evaluations
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, validation_dataset, validation_subset)
    """
    # Get smaller training set
    train_dataset = dataset['train'].shuffle(seed=seed).select(
        range(min(train_size, len(dataset['train'])))
    )
    
    # Full validation set
    validation_dataset = dataset['validation']
    
    # Create smaller validation subset if requested
    if validation_size is not None:
        validation_subset = dataset['validation'].shuffle(seed=seed).select(
            range(min(validation_size, len(dataset['validation'])))
        )
    else:
        validation_subset = validation_dataset
    
    print(f"Created dataset subsets:")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(validation_dataset)}")
    if validation_size is not None:
        print(f"  Validation subset examples: {len(validation_subset)}")
    
    return train_dataset, validation_dataset, validation_subset


def analyze_class_distribution(
    dataset: Dataset, 
    label_col: str = 'label',
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[int, int]:
    """
    Analyze and visualize class distribution in the dataset.
    
    Args:
        dataset: Dataset to analyze
        label_col: Name of the label column
        class_names: Optional list of class names for visualization
        save_path: Path to save the visualization
        
    Returns:
        Dictionary mapping class labels to counts
    """
    if label_col not in dataset.features:
        raise ValueError(f"Label column '{label_col}' not found in dataset")
    
    # Count labels
    labels = dataset[label_col]
    label_counts = pd.Series(labels).value_counts().sort_index()
    
    # Print statistics
    print("\nClass distribution:")
    total = len(labels)
    for label, count in label_counts.items():
        class_name = ""
        if class_names and label < len(class_names):
            class_name = f" ({class_names[label]})"
            
        print(f"  Label {label}{class_name}: {count} examples ({count/total*100:.1f}%)")
    
    # Calculate imbalance
    if len(label_counts) > 1:
        imbalance_ratio = label_counts.max() / label_counts.min()
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 1.5:
            print("Note: The dataset shows class imbalance. F1 score may be a more reliable metric than accuracy.")
        else:
            print("The dataset is relatively balanced.")
    
    # Create visualization
    if save_path is not None:
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        ax = sns.barplot(
            x=label_counts.index.astype(str), 
            y=label_counts.values, 
            palette='viridis'
        )
        
        # Add count and percentage labels
        for i, (label, count) in enumerate(label_counts.items()):
            ax.text(
                i, count/2, 
                f"{count}\n({count/total*100:.1f}%)", 
                ha='center', va='center', 
                color='white', fontweight='bold'
            )
        
        # Customize plot
        if class_names and len(class_names) == len(label_counts):
            plt.xticks(
                range(len(class_names)), 
                [f"{name} ({i})" for i, name in enumerate(class_names)]
            )
        
        plt.title('Class Distribution', fontsize=14)
        plt.xlabel('Class Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(save_path)
        plt.close()
        print(f"Class distribution visualization saved to {save_path}")
    
    return dict(label_counts)


def preprocess_function(examples, tokenizer, max_length=128):
    """
    Tokenize and prepare examples for the model.
    
    Args:
        examples: Examples to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    # Determine text field name
    text_field = "sentence" if "sentence" in examples else "text"
    
    return tokenizer(
        examples[text_field], 
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


def process_dataset(
    dataset: Dataset, 
    tokenizer: PreTrainedTokenizer, 
    max_length: int = 128,
    batch_size: int = 1000
) -> Dataset:
    """
    Tokenize and prepare dataset for training.
    
    Args:
        dataset: Dataset to process
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size for processing
        
    Returns:
        Processed dataset
    """
    # Apply tokenization
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        batch_size=batch_size,
        desc=f"Tokenizing dataset"
    )
    
    return tokenized_dataset


def format_for_pytorch(dataset: Dataset) -> Dataset:
    """
    Format a dataset for PyTorch by removing unnecessary columns 
    and renaming labels.
    
    Args:
        dataset: Hugging Face dataset
        
    Returns:
        Formatted dataset ready for PyTorch
    """
    # Make a copy to avoid modifying the original
    formatted_dataset = dataset
    
    # Remove unnecessary columns
    if "sentence" in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.remove_columns(["sentence"])
    if "idx" in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.remove_columns(["idx"])
    if "text" in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.remove_columns(["text"])
    
    # Rename label to labels for Trainer compatibility
    if "label" in formatted_dataset.column_names and "labels" not in formatted_dataset.column_names:
        formatted_dataset = formatted_dataset.rename_column("label", "labels")
    
    # Set format to PyTorch tensors
    formatted_dataset.set_format("torch")
    
    return formatted_dataset


def prepare_datasets(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    validation_subset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare all datasets for training and evaluation.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Full validation dataset
        validation_subset: Smaller validation subset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (processed_train, processed_validation, processed_validation_subset)
    """
    # Process all datasets
    print("Processing datasets...")
    
    processed_train = process_dataset(
        train_dataset, tokenizer, max_length
    )
    
    processed_validation = process_dataset(
        validation_dataset, tokenizer, max_length
    )
    
    # Process validation subset if it's different from the full validation set
    if validation_subset is validation_dataset:
        processed_validation_subset = processed_validation
    else:
        processed_validation_subset = process_dataset(
            validation_subset, tokenizer, max_length
        )
    
    # Format for PyTorch
    processed_train = format_for_pytorch(processed_train)
    processed_validation = format_for_pytorch(processed_validation)
    processed_validation_subset = format_for_pytorch(processed_validation_subset)
    
    return processed_train, processed_validation, processed_validation_subset


def examine_processed_example(dataset: Dataset, tokenizer: PreTrainedTokenizer, index: int = 0):
    """
    Examine a processed example to verify tokenization.
    
    Args:
        dataset: Processed dataset
        tokenizer: Tokenizer used to process the dataset
        index: Index of the example to examine
    """
    if len(dataset) <= index:
        print(f"Dataset only has {len(dataset)} examples, cannot access index {index}")
        return
    
    print("\nExamining processed example:")
    example = dataset[index]
    
    if 'input_ids' in example:
        print(f"Input shape: {example['input_ids'].shape}")
        # Decode the input IDs to verify tokenization
        decoded_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        print(f"Decoded text: {decoded_text}")
    else:
        print("No input_ids found in the example")
    
    if 'attention_mask' in example:
        print(f"Attention mask shape: {example['attention_mask'].shape}")
    
    if 'labels' in example:
        print(f"Label: {example['labels']}")
    elif 'label' in example:
        print(f"Label: {example['label']}")
    else:
        print("No label found in the example")


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Load dataset
    dataset = load_sst2_dataset()
    
    # Create subsets
    train_dataset, validation_dataset, validation_subset = create_data_subsets(
        dataset, train_size=1000, validation_size=200
    )
    
    # Analyze class distribution
    analyze_class_distribution(
        train_dataset, 
        class_names=["Negative", "Positive"],
        save_path="class_distribution.png"
    )
    
    # Process datasets
    processed_train, processed_validation, processed_validation_subset = prepare_datasets(
        train_dataset, validation_dataset, validation_subset, tokenizer
    )
    
    # Examine a processed example
    examine_processed_example(processed_train, tokenizer)
