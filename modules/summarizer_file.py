# modules/summarizer_file.py

from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
import torch

print("✅ summarizer_file.py loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t5_tokenizer = T5Tokenizer.from_pretrained("models/my_t5_model")
t5_model = T5ForConditionalGeneration.from_pretrained("models/my_t5_model").to(device)
t5_model.eval()

bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("models/bart-cnn-finetuned").to(device)
bart_model.eval()

def summarize_text_t5(text):
    input_ids = t5_tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True).to(device)
    output = t5_model.generate(input_ids, max_length=150, min_length=30)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

def summarize_text_bart(text):
    inputs = bart_tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(device)
    summary_ids = bart_model.generate(inputs['input_ids'], max_length=150, min_length=30, length_penalty=2.0)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
