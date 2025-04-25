# Parameter-Efficient Fine-Tuning for NLP Models

![PEFT Performance](results/figures/accuracy_comparison.png)

## Overview

This project demonstrates state-of-the-art Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt foundation models for sentiment analysis with minimal computational resources. We explore LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) approaches, along with extensive analysis of different configurations.

### Key Results

- **84% accuracy** with fine-tuned model (up from 43% with base model)
- Training only **0.23%** of model parameters (1.77 million vs. 67 million parameters)
- Memory usage reduction through quantization techniques
- Comprehensive comparison of different LoRA configurations

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [References](#references)

## Background

Large language models have revolutionized natural language processing, but adapting them to specific tasks traditionally requires fine-tuning billions of parameters, demanding substantial computational resources. Parameter-Efficient Fine-Tuning (PEFT) addresses this challenge by updating only a small subset of parameters while freezing the pre-trained weights.

This project focuses on:

- **LoRA (Low-Rank Adaptation)** - Uses low-rank matrices to adapt pre-trained models efficiently
- **QLoRA (Quantized LoRA)** - Combines quantization with LoRA for even greater memory efficiency
- **Configuration optimization** - Experiments with different ranks, target modules, and hyperparameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/parameter-efficient-fine-tuning.git
cd parameter-efficient-fine-tuning

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `PEFT_Project.ipynb` - Jupyter notebook with complete implementation
- `src/` - Python modules with reusable code components
- `results/` - Analysis results and visualizations
- `models/` - Directory for saved model weights
- `docs/` - Additional documentation on techniques and usage

## Usage

### Running the Notebook

```bash
jupyter notebook PEFT_Project.ipynb
```

### Fine-tuning with LoRA

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,                        # Rank
    lora_alpha=32,               # Alpha parameter
    lora_dropout=0.1,            # Dropout probability
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
    bias="none",
)

# Create PEFT model
peft_model = get_peft_model(model, lora_config)

# Check trainable parameters
peft_model.print_trainable_parameters()
```

### Inference with Saved Model

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification, PeftConfig

# Load saved PEFT model
peft_model_path = "./models/lora_model"
peft_config = PeftConfig.from_pretrained(peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
model = AutoPeftModelForSequenceClassification.from_pretrained(peft_model_path)

# Sentiment prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    return "Positive" if prediction == 1 else "Negative"

# Example usage
predict_sentiment("I absolutely loved this product!")  # Output: Positive
```

## Results

### Performance Comparison

| Model | Accuracy | F1 Score | Parameters Trained | Memory Usage |
|-------|----------|----------|-------------------|--------------|
| Base Model | 43.2% | 0.34 | 100% | 3076.95 MB |
| LoRA (r=16) | 87.3% | 0.87 | 0.23% | 3081.46 MB |
| QLoRA | 84.5% | 0.85 | 0.11% | 2701.94 MB |
| High Rank (r=32) | 84.2% | 0.84 | 0.34% | 3089.12 MB |
| Query-Only | 83.0% | 0.83 | 0.06% | 3079.32 MB |

### Parameter Efficiency

![Parameter Efficiency](results/figures/parameter_efficiency.png)

PEFT techniques achieved accuracy improvements of over 40 percentage points while training less than 0.25% of the parameters, demonstrating remarkable efficiency.

### Memory Usage

QLoRA reduced memory usage by approximately 12% compared to the base model while maintaining strong performance, making it suitable for resource-constrained environments.

## Technical Details

### Dataset

The SST-2 (Stanford Sentiment Treebank) dataset was used for sentiment analysis, consisting of movie reviews labeled as positive or negative.

### Hyperparameters

- Base model: `distilbert-base-uncased`
- LoRA rank (r): Experiments with 4, 16, and 32
- Alpha values: 16-64
- Dropout: 0.05-0.3
- Learning rates: 1e-4 to 5e-4
- Training epochs: 3

### Key Findings

1. Higher rank values (r=32) provided marginal improvements over lower ranks but required more parameters
2. Training only query projections (Query-Only) was most parameter-efficient but had lower performance
3. Including bias parameters in training provided small performance improvements
4. QLoRA achieved significant memory savings with minimal performance impact

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment](https://arxiv.org/abs/2110.07602)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)

## License

This project is licensed under the AGPL License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for their transformers and PEFT libraries
- GLUE benchmark for providing the SST-2 dataset
