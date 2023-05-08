import torch
import transformers
import wget
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

url = 'http://gutenberg.net.au/ebooks02/0200041.txt'  # the great gatsby full text
filename = wget.download(url)
print(filename)

tmp_text = []
with open(filename, errors="replace") as f:
    for line in f:
        tmp_text.append(line)
        if len(tmp_text) > 750:
            break
text = "".join(tmp_text)
text = text.replace("THE END", "EPILOGUE")
text = text.replace("Project Gutenberg Australia", "")
print(text[:-100])



# non-triton ver.
model_name = 'mosaicml/mpt-7b-storywriter'
config = transformers.AutoConfig.from_pretrained(
  model_name,
  trust_remote_code=True
)
#config.update({"max_seq_len": 83968})
config.update({"max_seq_len": 8396})
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
  model_name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True
)
model.to(device='cuda:0')

input_ids = tokenizer(text, return_tensors="pt").input_ids
input_ids = input_ids.to(model.device)
max_new_tokens = 120
temperature = 0.1
generate_kwargs = dict(
    input_ids=input_ids,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    do_sample=temperature > 0.0,
    top_p=0.01,
    top_k=1,
    repetition_penalty=1.0,
)

output_text = model.generate(**generate_kwargs)
print(tokenizer.decode(output_text[0], skip_special_tokens=True))
