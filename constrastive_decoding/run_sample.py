from transformers import AutoTokenizer, set_seed
from accelerate.test_utils.testing import get_backend

from contrastive_decoding import (
    hijack_generation,
    ContrastiveLlamaForCausalLM,
)
import torch



messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
hijack_generation()

main_model = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(main_model)
model = ContrastiveLlamaForCausalLM.from_pretrained(main_model, torch_dtype=torch.bfloat16)
assistant_model = ContrastiveLlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16
)
device, _, _ = (
    get_backend()
)  # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model = model.to(device)
assistant_model = assistant_model.to(device)
set_seed(42)

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)

vanilla_output = model.generate(tokenized_chat, do_sample=False, max_new_tokens=50)
print(tokenizer.decode(vanilla_output[0]))
print("--------------------------------")
outputs = model.generate_contrastive(
    tokenized_chat, do_sample=False, max_new_tokens=50, assistant_model=assistant_model
)
print(tokenizer.decode(outputs[0]))

