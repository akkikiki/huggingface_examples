from mlx_lm import load, generate
import mlx.core as mx

def load_llama_mlx(model_name="mlx-community/Llama-2-7b-chat-mlx"):
    """
    Load Llama model using MLX for efficient inference on Mac.
    
    Args:
        model_name (str): Name of the MLX-compatible Llama model to load
        
    Returns:
        model: The loaded MLX model
        tokenizer: The model's tokenizer
    """
    model, tokenizer = load(model_name)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_tokens=100, temperature=0.7):
    """
    Generate text using the MLX Llama model.
    
    Args:
        prompt (str): Input text to generate from
        model: The loaded MLX model
        tokenizer: The model's tokenizer
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature (higher = more random)
        
    Returns:
        str: Generated text
    """
    tokens = generate(
        prompt,
        model,
        tokenizer,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return tokenizer.decode(tokens)

def main():
    # Example usage
    model, tokenizer = load_llama_mlx()
    
    prompt = "Write a short poem about artificial intelligence:"
    print(f"Prompt: {prompt}")
    
    response = generate_text(prompt, model, tokenizer)
    print(f"\nGenerated Response:\n{response}")

if __name__ == "__main__":
    main()
