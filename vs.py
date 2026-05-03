from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

# Load from relative model folder
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Load tokenizer & model
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Define labels in the same order as training
labels = ["anxiety", "depression", "lonely", "mental_health", "suicide_watch"]

# Example text
text = "I feel so alone and hopeless these days."

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
    outputs = model(**inputs)

pred = torch.argmax(outputs.logits, dim=1).item()
print(f"Predicted Label: {labels[pred]}")
