# Ref: https://github.com/Blaizzy/mlx-vlm
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model_path = "mlx-community/paligemma2-3b-ft-docci-448-8bit"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
image = ["https://akkikiki.github.io/assets/img/publication_preview/emnlp2023_bias_preview-1400.webp"]
prompt = "Transcribe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=len(image)
)

# Generate output
output = generate(model, processor, image, formatted_prompt, verbose=False, max_tokens=400)
print(output)
