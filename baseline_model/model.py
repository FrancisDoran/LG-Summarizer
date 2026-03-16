"""

This is a baseline inference run of the BART Language Model.

Using facebook/bart-base (pre-trained, not fine-tuned for summarization).

Model was loaded via Hugging Face.

"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#Set to CUDA gpu if available, else -> CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "facebook/bart-large-cnn"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float32,
)

text = (
    "The tower is 324 metres (1,063 ft) tall, about the same height as an "
    "81-storey building, and the tallest structure in Paris. Its base is square, "
    "measuring 125 metres (410 ft) on each side. During its construction, the "
    "Eiffel Tower surpassed the Washington Monument to become the tallest "
    "man-made structure in the world, a title it held for 41 years until the "
    "Chrysler Building in New York City was finished in 1930. It was the first "
    "structure to reach a height of 300 metres. Due to the addition of a "
    "broadcasting aerial at the top of the tower in 1957, it is now taller than "
    "the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the "
    "Eiffel Tower is the second tallest free-standing structure in France after "
    "the Millau Viaduct."
)

inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)

# ensure no gradients are calculated or applied (we are doing inference, not training)
with torch.no_grad():
    outputs = model.generate(**inputs)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

print(model.__class__.__name__)
print(model.config.activation_function)
