# Adjusted Loss Module for Discrete Flow Matching

This repository contains an adjusted version of the `MixturePathGeneralizedKL` loss module specifically designed to work with discrete flow matching for language models.

## Key Adjustments Made

### 1. **Constructor Flexibility**
- **Original**: Required a `MixtureDiscreteProbPath` parameter
- **Adjusted**: Made the `path` parameter optional with a default linear scheduler
- **Benefit**: Easier integration with existing training code

### 2. **Input Shape Handling**
- **Original**: Expected specific tensor dimensions (batch, d, K)
- **Adjusted**: Handles language model tensors (batch, seq_len, vocab_size)
- **Benefit**: Works directly with transformer model outputs

### 3. **Masking Support**
- **Original**: No explicit support for masked positions
- **Adjusted**: Properly handles `-100` values (standard PyTorch ignore index)
- **Benefit**: Compatible with standard language modeling practices

### 4. **Batch Processing**
- **Original**: Simple batch processing
- **Adjusted**: Efficient handling of variable-length sequences and masking
- **Benefit**: Better memory usage and performance

### 5. **Additional Loss Function**
- **New**: Added `SimplifiedFlowLoss` as a fallback option
- **Benefit**: Easier debugging and comparison with standard cross-entropy

## Files Included

### `loss.py`
The main loss module containing:
- `MixturePathGeneralizedKL`: The adjusted generalized KL loss
- `MixtureDiscreteProbPath`: Simple probability path implementation
- `SchedulerOutput`: Container for scheduler outputs
- `SimplifiedFlowLoss`: Simplified flow-based loss for comparison

### `train_fixed.py`
Updated training script that:
- Properly integrates the adjusted loss module
- Supports both flow loss types
- Includes comprehensive evaluation and logging
- Uses proper tensor shapes and masking

### `example_usage.py`
Demonstration script showing:
- How to use both loss functions
- Testing with dummy data
- Gradient flow verification
- Comparison with standard cross-entropy loss

## Usage Examples

### Basic Usage

```python
from loss import MixturePathGeneralizedKL, MixtureDiscreteProbPath

# Create the loss function
path = MixtureDiscreteProbPath()
loss_fn = MixturePathGeneralizedKL(path=path, reduction="mean")

# Use in training loop
loss = loss_fn(logits, target, noised_input, time_t)
```

### With Custom Scheduler

```python
def custom_scheduler(t):
    from loss import SchedulerOutput
    alpha_t = (1 - t) ** 2  # Quadratic schedule
    d_alpha_t = -2 * (1 - t)
    return SchedulerOutput(alpha_t, d_alpha_t)

path = MixtureDiscreteProbPath(scheduler_fn=custom_scheduler)
loss_fn = MixturePathGeneralizedKL(path=path, reduction="mean")
```

### Simplified Version

```python
from loss import SimplifiedFlowLoss

# Easier to use, time-weighted cross-entropy
loss_fn = SimplifiedFlowLoss(reduction="mean")
loss = loss_fn(logits, target, noised_input, time_t)
```

## Integration with Training Code

The adjusted loss module integrates seamlessly with the provided training code:

```python
# In the training loop
if use_flow_loss:
    path = MixtureDiscreteProbPath()
    loss_fn = MixturePathGeneralizedKL(path=path, reduction="mean")
else:
    loss_fn = SimplifiedFlowLoss(reduction="mean")

# Calculate loss
loss = loss_fn(logits, target, noised_input_ids, time_t)
```

## Key Benefits

1. **Drop-in Replacement**: Minimal changes needed to existing training code
2. **Proper Masking**: Handles padding and ignore tokens correctly
3. **Memory Efficient**: Optimized for large vocabulary sizes
4. **Flexible Scheduling**: Support for custom noise schedules
5. **Gradient Stable**: Proper gradient flow for training stability

## Testing

Run the example script to verify everything works:

```bash
python example_usage.py
```

This will test all loss functions, verify gradient flow, and compare with standard cross-entropy loss.

## Mathematical Background

The generalized KL loss for discrete flow matching is given by:

$$\ell_i(x_1, x_t, t) = -\frac{\dot{\kappa}_t}{1-\kappa_t} \left[  p_{1|t}(x_t^i|x_t) -\delta_{x^i_1}(x_t^i) + (1-\delta_{x^i_1}(x_t^i))\left(\log p_{1|t}(x_1^i|x_t)\right)\right]$$

Where:
- $\kappa_t$ is the noise schedule
- $p_{1|t}$ is the model's predicted distribution
- $\delta$ is the Dirac delta function
- The loss is computed only on valid (non-masked) positions

## Compatibility

- PyTorch >= 1.9
- Compatible with Hugging Face transformers
- Works with accelerate for distributed training
- Supports mixed precision training
