import transformers
import wget
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# non-triton ver.
model_name = 'mosaicml/mpt-7b-storywriter'
config = transformers.AutoConfig.from_pretrained(
  model_name,
  trust_remote_code=True
)
config.update({"max_seq_len": 83968})
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
  model_name,
  config=config,
  trust_remote_code=True
)
#model.to(device='cuda:0')

url = 'http://gutenberg.net.au/ebooks02/0200041.txt'  # the great gatsby full text
filename = wget.download(url)
with open(filename) as f:
    text = f.readlines()
text = text.replace("THE END", "EPILOGUE")
text = text.replace("Project Gutenberg Australia", "")
print(text[:-100])

input_ids = tokenizer(text, return_tensors="pt").input_ids
input_ids = input_ids.to(m.device)
streamer = TextIteratorStreamer(tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
generate_kwargs = dict(
    input_ids=input_ids,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    do_sample=temperature > 0.0,
    top_p=top_p,
    top_k=top_k,
    repetition_penalty=repetition_penalty,
    streamer=streamer,
    stopping_criteria=StoppingCriteriaList([stop]),
)

output_text = m.generate(**generate_kwargs)
print(output_text)
