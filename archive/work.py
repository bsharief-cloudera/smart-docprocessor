#pip install -q transformers torch accelerate autoawq 
#!pip install -q  --upgrade torchvision

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Karen_TheEditor_V2_STRICT_Mistral_7B-AWQ")

# Load the model with AWQ quantization
model = AutoModelForCausalLM.from_pretrained("TheBloke/Karen_TheEditor_V2_STRICT_Mistral_7B-AWQ", torch_dtype=torch.float16)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the generation pipeline with the recommended settings
editor = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)


# Example input text
text_to_correct = "This is a smaple sentence with some errors.We is misspelled few word"

# ChatML formatted input
input_text = f"<|system|>Please correct the spelling and grammatical errors in the following text.<|end|>\n" \
             f"<|user|>{text_to_correct}<|end|>\n" \
             f"<|assistant|>"

# Generate text using the recommended settings
output = editor(input_text, max_length=512, temperature=0.7, top_p=0.1, top_k=40, repetition_penalty=1.18, num_return_sequences=1)

# Print the result
print("Original text:\n", text_to_correct)
print("Corrected text:\n", output[0]['generated_text'].replace("<|assistant|>", "").strip())