import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.utils import generate_step

# Load the model
model_path = "mlx-community/Llama-2-7b-chat-mlx"
model, tokenizer = load(model_path)

# Prepare the prompt
prompt = "What is the capital of France?"

# Tokenize the prompt
tokens = tokenizer.encode(prompt)
tokens = mx.array([tokens])

# Generate text
response = generate(model, tokenizer, prompt=prompt, max_tokens=100, temp=0.7)
print("\nResponse:", response)

# Example of interactive chat
def chat(prompt, history=[]):
    history.append({"role": "user", "content": prompt})
    chat_text = tokenizer.apply_chat_template(history, tokenize=False)
    
    response = generate(model, tokenizer, prompt=chat_text, max_tokens=100, temp=0.7)
    history.append({"role": "assistant", "content": response})
    return response, history

# Example chat interaction
chat_history = []
response, chat_history = chat("Hello! How are you?")
print("\nChat Response:", response)
