import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
from accelerate import init_empty_weights, infer_auto_device_map

model_id = "google/flan-ul2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = T5Config.from_pretrained(model_id)

max_memory={i: "16GiB" for i in range(4)}  # Assume 4 GPUs
max_memory[0] = "10GiB"  # to fit lm_head to the same device as the inputs

with init_empty_weights():
    model = T5ForConditionalGeneration(config)
    device_map = infer_auto_device_map(model, no_split_module_classes=["T5Block"], dtype=torch.float16, max_memory=max_memory)
device_map['lm_head'] = device_map["decoder.embed_tokens"]

model = T5ForConditionalGeneration.from_pretrained(model_id, device_map=device_map, load_in_8bit=True)

input_text = "Answer the following question by reasoning step by step. The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids, max_length=100)
print(tokenizer.decode(outputs[0]))
