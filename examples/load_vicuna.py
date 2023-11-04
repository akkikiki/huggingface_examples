import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "lmsys/vicuna-13b-v1.3"
config_opt = AutoConfig.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    offload_folder="offload",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
prompt = "All you need is love. But a little chocolate now "
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(input_ids=inputs["input_ids"].cuda(), max_new_tokens=30)
print(" output prediction: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
