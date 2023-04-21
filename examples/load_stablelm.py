from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-base-alpha-7b")
model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-base-alpha-7b")
model.half().cuda()

inputs = tokenizer("What's your mood today?", return_tensors="pt").to("cuda")
tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.7,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
