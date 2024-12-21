# huggingface_examples

Examples of Huggingface Transformers, Accelerate, and PEFT code mainly for my educational purpose.
Hope it helps somebody :)


## Environment Setup

```
conda create -n hf-examples python=3.10
conda activate hf-examples
pip install -r requirements.txt
```

Don't forget to set `export HF_HOME=YOUR_CACHE_DIR` when using volume storage.

## MLX Examples

MLX is Apple's machine learning framework optimized for Apple Silicon. Here are examples of loading and using different models with MLX:

### Basic Model Loading

```python
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load a model using MLX backend
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float16")

# Convert input to MLX format
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="np")
input_ids = mx.array(inputs["input_ids"])

# Generate text
outputs = model.generate(input_ids, max_length=50)
result = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
```

### Loading Custom Weights

```python
import mlx.core as mx
from mlx_lm import load, generate

# Load a model with custom weights
model, tokenizer = load("mlx-community/Mistral-7B-v0.1-mlx")

# Generate text
prompt = "Write a story about"
result = generate(model, tokenizer, prompt=prompt, max_tokens=100)
```

### Fine-tuning Example

```python
import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def __call__(self, x):
        return self.linear(x)

# Initialize model and optimizer
model = SimpleModel()
optimizer = Adam(learning_rate=0.001)

# Training loop example
def train_step(model, x, y):
    def loss_fn(model, x, y):
        y_pred = model(x)
        return nn.losses.cross_entropy(y_pred, y)
    
    loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss
```

### Loading Vision Models

```python
from transformers import AutoProcessor, AutoModelForImageClassification
import mlx.core as mx
from PIL import Image

# Load vision model
model_name = "apple/mobilevit-small"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Process image
image = Image.open("example.jpg")
inputs = processor(images=image, return_tensors="np")
pixel_values = mx.array(inputs["pixel_values"])

# Get predictions
outputs = model(pixel_values)
predicted_label = outputs.logits.argmax(-1).item()
```

### Memory-Efficient Loading

```python
import mlx.core as mx
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model with memory optimizations
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="float16",
    quantization_config={"bits": 4}  # 4-bit quantization
)

# Use model with reduced memory footprint
text = "Translate to French:"
inputs = tokenizer(text, return_tensors="np")
input_ids = mx.array(inputs["input_ids"])
outputs = model.generate(input_ids, max_length=50)
```

These examples demonstrate various ways to use MLX for different machine learning tasks. MLX is particularly efficient on Apple Silicon hardware and provides a familiar API for those coming from PyTorch or other frameworks.

Note: Make sure you have the required dependencies installed:
```bash
pip install mlx transformers mlx-lm Pillow
```
