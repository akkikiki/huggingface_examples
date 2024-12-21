# huggingface_examples

Examples of Huggingface Transformers, Accelerate, PEFT, and MLX code mainly for educational purpose.
Hope it helps somebody :)

## MLX Examples (Mac/Apple Silicon)

The `mlx` directory contains examples for running models using Apple's MLX framework:

- `llama_example.py`: Run Llama 2 models using MLX
- `paligemma2_example.py`: Run PaLI-GEMMA 2 for vision-language tasks

These examples require an Apple Silicon Mac and the MLX framework installed.


## Environment Setup

```
conda create -n hf-examples python=3.10
pip install -r requirements.txt

```

Don't forget to set `export HF_HOME=YOUR_CACHE_DIR` when using volume storage.
