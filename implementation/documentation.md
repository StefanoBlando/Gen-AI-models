# Parameter-Efficient Fine-Tuning (PEFT) Techniques

This document provides an overview of the Parameter-Efficient Fine-Tuning techniques used in this project, focusing on LoRA and QLoRA.

## Why PEFT?

Traditional fine-tuning of large pre-trained models has several limitations:

1. **High computational demands**: Fine-tuning all parameters requires substantial GPU memory and computing power
2. **Storage costs**: Storing multiple fine-tuned versions of large models is expensive
3. **Catastrophic forgetting**: Full fine-tuning can cause the model to "forget" previously learned knowledge

Parameter-efficient fine-tuning addresses these challenges by modifying only a small subset of parameters, typically less than 1%, while keeping most of the pre-trained weights frozen.

## LoRA: Low-Rank Adaptation

### Key Concept

LoRA performs model adaptation by inserting low-rank decomposition matrices into the model architecture. Instead of fine-tuning all parameters of a weight matrix W, LoRA freezes the pre-trained weights and injects trainable rank decomposition matrices A and B such that:

```
W' = W + BA
```

where:
- W' is the effective adapted weight matrix
- W is the frozen pre-trained weight matrix
- B and A are low-rank matrices (r << d, where d is the original dimension)

### Mathematical Details

For a weight matrix W ∈ ℝᵐˣⁿ, LoRA constrains its update to:
- A ∈ ℝʳˣⁿ 
- B ∈ ℝᵐˣʳ
- Where r is the low rank (typically 4-32)

The number of trainable parameters becomes r(m+n) instead of mn, which can be orders of magnitude smaller.

### Key Hyperparameters

- **r (rank)**: Controls the rank of the decomposition. Higher values provide more modeling capacity but require more parameters.
- **alpha**: Scaling factor for the LoRA adaptation (typically set to 2x the rank value).
- **dropout**: Applied to the LoRA activations, helps with generalization.
- **target_modules**: Which model components to apply LoRA to (e.g., attention query, key, value projections).
- **bias**: Whether to train bias parameters ("none", "all", or "lora_only").

### Initialization

LoRA typically initializes matrix A to be random Gaussian and matrix B to be zero, meaning the model starts from the pre-trained state and gradually adapts as training proceeds.

## QLoRA: Quantized Low-Rank Adaptation

QLoRA extends LoRA by applying quantization techniques to further reduce memory requirements. The key components of QLoRA include:

### 4-bit Quantization

The base model weights are quantized to 4-bit precision (typically using NormalFloat4 quantization), drastically reducing memory footprint for the frozen parameters.

### Double Quantization

QLoRA applies a second round of quantization to the quantization constants themselves, further reducing memory usage.

### Paged Optimizers

Uses paged optimizers to manage GPU memory more efficiently by offloading optimizer states to CPU when not in use.

### Advantages of QLoRA

1. **Extreme memory efficiency**: Can fine-tune models on consumer hardware that would otherwise be impossible
2. **Minimal performance degradation**: Despite aggressive quantization, performance remains comparable to full-precision LoRA
3. **More efficient inference**: Quantized models require less memory during inference

## Comparison in Our Project

In our experiments, we found:

1. **LoRA**: Achieved 87.3% accuracy (compared to base model's 43.2%) while training only 0.23% of parameters
2. **QLoRA**: Achieved 84.5% accuracy while further reducing memory usage by approximately 12%
3. **Configuration variants**: Different LoRA configurations showed distinct tradeoffs between parameter efficiency and performance

## Choosing the Right Configuration

Based on our experiments, consider these guidelines:

1. **Limited hardware resources**: Use QLoRA with lower rank values (r=4 or r=8)
2. **Maximum performance**: Use LoRA with higher rank values (r=16 or r=32)
3. **Extreme parameter efficiency**: Target only query projections, though with some performance tradeoff
4. **General purpose**: LoRA with r=16, targeting query, key, value, and output projections provides a good balance

## References

- [LoRA Paper: "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper: "QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314)
- [PEFT Library Documentation](https://huggingface.co/docs/peft/index)
