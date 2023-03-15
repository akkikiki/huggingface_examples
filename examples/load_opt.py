import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map

#model_id = "facebook/opt-30b"
model_id = "facebook/opt-66b"
config_opt = AutoConfig.from_pretrained(model_id)
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

max_memory={i: "6GiB" for i in range(8)}  # Assume 8 GPUs, 6GiB is an arbitrary value which did not cause OOM error
max_memory["cpu"] = "200GiB"  # offloading to CPU with maximum of 200GB memory usage

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config_opt)
    device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], dtype=torch.float16, max_memory=max_memory)

device_map['lm_head'] = device_map["model.decoder.embed_tokens"]
print(device_map)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map=device_map,
    offload_folder="offload",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
prompt = "All you need is love. But a little chocolate now "
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(input_ids=inputs["input_ids"].cuda(), max_new_tokens=30)
print(" output prediction: ", tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))
