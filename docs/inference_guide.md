# Inference Guide: Using Fine-Tuned PEFT Models

This guide provides instructions for loading and using the fine-tuned PEFT models for inference in various scenarios.

## Basic Inference Setup

### Loading a LoRA Model

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

### Simple Sentiment Analysis Function

```python
import torch

def predict_sentiment(text, model, tokenizer, return_scores=False):
    """
    Predict sentiment for a given text.
    
    Args:
        text: Input text string or list of strings
        model: LoRA fine-tuned model
        tokenizer: Tokenizer for the model
        return_scores: Whether to return confidence scores
        
    Returns:
        If return_scores=False: "Positive" or "Negative"
        If return_scores=True: Dict with prediction and confidence scores
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
        max_length=128
    )
    
    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
```

### Example Usage

```python
# Single prediction
sentiment = predict_sentiment("I absolutely loved this product!", model, tokenizer)
print(sentiment)  # Output: "Positive"

# Prediction with confidence scores
result = predict_sentiment("The service was terrible.", model, tokenizer, return_scores=True)
print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")

# Batch prediction
texts = [
    "This is amazing!",
    "I'm not happy with my purchase.",
    "It was okay, not great but not bad either."
]
predictions = predict_sentiment(texts, model, tokenizer)
for text, pred in zip(texts, predictions):
    print(f"{text} => {pred}")
```

## Advanced Usage

### Merging LoRA Weights with Base Model

For deployment scenarios where inference efficiency is critical, you can merge the LoRA weights with the base model:

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=2
)

# Load PEFT model
peft_model = PeftModel.from_pretrained(base_model, model_path)

# Merge weights
merged_model = peft_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./models/merged_model")
tokenizer.save_pretrained("./models/merged_model")
```

The merged model can be loaded like a regular Hugging Face model:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

merged_model = AutoModelForSequenceClassification.from_pretrained("./models/merged_model")
tokenizer = AutoTokenizer.from_pretrained("./models/merged_model")
```

### Using QLoRA Models

QLoRA models are loaded the same way as LoRA models, but need special handling for quantization:

```python
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the model with quantization
model = AutoPeftModelForSequenceClassification.from_pretrained(
    "./models/qlora_model",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### Serving Models with FastAPI

For production deployments, you can wrap the model in a FastAPI service:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class SentimentRequest(BaseModel):
    texts: List[str]
    return_scores: Optional[bool] = False

class SentimentResponse(BaseModel):
    results: List[dict]

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    try:
        results = predict_sentiment(
            request.texts, 
            model, 
            tokenizer, 
            return_scores=request.return_scores
        )
        
        # Format results for response
        if not request.return_scores:
            results = [{"prediction": r} for r in results]
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Optimizing Inference

### GPU Inference

For optimal GPU inference performance:

```python
# Move model to GPU
model = model.to("cuda")

# Use CUDA for tensors
inputs = {k: v.to("cuda") for k, v in inputs.items()}

# Use torch.compile for PyTorch 2.0+ (can give 1.5-3x speedup)
if hasattr(torch, 'compile'):
    model = torch.compile(model)
```

### CPU Inference

For CPU deployment:

```python
# Move model to CPU
model = model.to("cpu")

# Optimize for CPU inference
model = model.eval()

# Optionally convert to ONNX for faster CPU inference
import onnxruntime as ort
from transformers.onnx import export

# Export model to ONNX
onnx_path = "./models/model.onnx"
export(
    preprocessor=tokenizer,
    model=model,
    output=onnx_path,
    opset=13,
    input_names=["input_ids", "attention_mask"]
)

# Load with ONNX Runtime
ort_session = ort.InferenceSession(onnx_path)

# Inference function for ONNX model
def predict_sentiment_onnx(text, session, tokenizer):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Convert to numpy for ONNX Runtime
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }
    
    # Run inference
    ort_outputs = session.run(None, ort_inputs)
    
    # Process outputs
    logits = ort_outputs[0]
    probabilities = softmax(logits, axis=1)  # Use numpy softmax
    predictions = np.argmax(logits, axis=1)
    
    # Return prediction
    return "Positive" if predictions[0] == 1 else "Negative"
```

### Batch Processing for Large Datasets

For processing large datasets efficiently:

```python
import pandas as pd
from tqdm import tqdm

def batch_predict(texts, model, tokenizer, batch_size=32):
    """Process a large number of texts in batches"""
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        batch_results = predict_sentiment(batch, model, tokenizer, return_scores=True)
        results.extend(batch_results)
    
    return results

# Example with a CSV file
df = pd.read_csv("reviews.csv")
predictions = batch_predict(df["text"].tolist(), model, tokenizer)

# Add predictions to the dataframe
df["sentiment"] = [p["prediction"] for p in predictions]
df["confidence"] = [p["confidence"] for p in predictions]
df.to_csv("reviews_with_sentiment.csv", index=False)
```

## Model Interpretability

Understanding why a model made a particular prediction:

```python
from transformers import pipeline
from transformers_interpret import SequenceClassificationExplainer

# Create explainer
explainer = SequenceClassificationExplainer(
    model, 
    tokenizer
)

# Get attribution scores for a prediction
word_attributions = explainer("This movie was great but too long")

# Visualize attributions
explainer.visualize()

# Print attributions
for word, attribution in word_attributions:
    print(f"{word}: {attribution:.4f}")
```

## Gradio Web Interface

Create a simple web interface for your model:

```python
import gradio as gr

def predict_with_gradio(text):
    result = predict_sentiment(text, model, tokenizer, return_scores=True)
    
    # Format output for display
    label = result["prediction"]
    pos_score = result["probabilities"]["Positive"]
    neg_score = result["probabilities"]["Negative"]
    
    # Return label and confidence scores
    return label, {
        "Positive": float(pos_score),
        "Negative": float(neg_score)
    }

# Create Gradio interface
demo = gr.Interface(
    fn=predict_with_gradio,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs=[
        gr.Label(label="Sentiment"),
        gr.Label(label="Confidence")
    ],
    title="Sentiment Analysis with PEFT",
    description="Analyze the sentiment of text using a parameter-efficient fine-tuned model"
)

# Launch the interface
demo.launch()
```

## Performance Benchmarking

Benchmark your model's inference performance:

```python
import time
import numpy as np

def benchmark_inference(model, tokenizer, batch_size=1, num_iterations=100, seq_length=128):
    """Benchmark inference speed"""
    # Generate random text of specified length
    random_text = "This is a test " * (seq_length // 4)
    texts = [random_text] * batch_size
    
    # Warmup
    for _ in range(10):
        _ = predict_sentiment(texts, model, tokenizer)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = predict_sentiment(texts, model, tokenizer)
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_iterations
    avg_time_per_sample = avg_time_per_batch / batch_size
    samples_per_second = batch_size / avg_time_per_batch
    
    return {
        "batch_size": batch_size,
        "samples_per_second": samples_per_second,
        "avg_time_per_sample_ms": avg_time_per_sample * 1000,
        "avg_time_per_batch_ms": avg_time_per_batch * 1000
    }

# Run benchmark with different batch sizes
batch_sizes = [1, 4, 8, 16, 32]
results = []

for bs in batch_sizes:
    print(f"Benchmarking with batch size {bs}...")
    result = benchmark_inference(model, tokenizer, batch_size=bs)
    results.append(result)
    print(f"  Samples/second: {result['samples_per_second']:.2f}")
    print(f"  Time/sample: {result['avg_time_per_sample_ms']:.2f} ms")

# Display results as a table
import pandas as pd
pd.DataFrame(results).set_index("batch_size")
```

## Common Issues and Solutions

### Issue: Model outputs are different after loading

**Solution**: Ensure your tokenization process is identical between training and inference:

```python
# Make sure truncation and padding settings match
inputs = tokenizer(
    text, 
    truncation=True,               # Must match training
    padding="max_length",          # Must match training
    max_length=128,                # Must match training
    return_tensors="pt"
)
```

### Issue: Low confidence predictions

**Solution**: Consider implementing a threshold for predictions:

```python
def predict_with_threshold(text, model, tokenizer, threshold=0.7):
    result = predict_sentiment(text, model, tokenizer, return_scores=True)
    
    if result["confidence"] >= threshold:
        return result["prediction"]
    else:
        return "Uncertain"
```

### Issue: CUDA out of memory errors

**Solution**: Reduce batch size or use CPU offloading:

```python
# Use device_map for automatic offloading
model = AutoPeftModelForSequenceClassification.from_pretrained(
    model_path,
    device_map="auto"
)

# Or manually handle batches
def process_large_batch(texts, batch_size=8):
    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results = predict_sentiment(batch, model, tokenizer)
        all_results.extend(results)
        # Clear CUDA cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_results
```

## Conclusion

This guide covered the basics of using PEFT models for inference across various deployment scenarios. The main advantages of PEFT models for inference are:

1. **Smaller storage footprint**: LoRA adapters are much smaller than full fine-tuned models
2. **Flexibility**: The same base model can be adapted for multiple tasks with different adapters
3. **Efficiency**: QLoRA models provide additional memory savings during inference

For production deployments, consider:

1. Merging weights for maximum inference speed
2. Using quantization for reduced memory footprint
3. Implementing batching for processing large datasets
4. Adding proper error handling and fallback mechanisms

By leveraging these techniques, PEFT models can be efficiently deployed in a wide range of production environments.
