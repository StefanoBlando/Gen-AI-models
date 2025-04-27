# Model Weights

This directory is for storing saved model weights.

## Usage

When you run the notebook, model weights will be saved in this directory.
These files aren't tracked in git due to their size.

## Directory Structure

The notebook creates the following subdirectories:

- `lora_model/`: Contains the LoRA adapter weights for the main model
- `qlora_model/`: Contains the QLoRA adapter weights (if QLoRA implementation is enabled)
- `variant_*/`: Contains adapter weights for different LoRA configurations being tested
- `best_model/`: Contains the best-performing model weights identified during experimentation

## Loading Models

To load a saved model, see the inference examples in `docs/inference_guide.md`.

Example code for loading a saved PEFT model:

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForSequenceClassification, PeftConfig

# Path to your saved model
model_path = "./models/lora_model"

# Load the PEFT configuration
peft_config = PeftConfig.from_pretrained(model_path)

# Load the tokenizer based on the base model specified in the config
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the model
model = AutoPeftModelForSequenceClassification.from_pretrained(model_path)
```
