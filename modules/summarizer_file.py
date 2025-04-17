# modules/summarizer.py
print("summarizer.py loaded ✅")

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ensure the model is correctly loaded
tokenizer = T5Tokenizer.from_pretrained("models/my_t5_model")
model = T5ForConditionalGeneration.from_pretrained("models/my_t5_model")

# Define the summarize function
def summarize_text(text):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True)
    output = model.generate(input_ids, max_length=150, min_length=30)
    return tokenizer.decode(output[0], skip_special_tokens=True)
