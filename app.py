import gradio as gr
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import os

# --- Load fine-tuned model (relative path so it works from anywhere) ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

# Correct label order (matches your fine-tuned model)
labels = ["anxiety", "depression", "lonely", "mental_health", "suicide_watch"]

# --- Maintain a log list ---
prediction_log = []

# --- Prediction function ---
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Predicted label
    pred_index = torch.argmax(probs).item()
    pred_label = labels[pred_index]
    confidence = float(probs[0][pred_index])

    # Probabilities as percentages
    all_probs = {labels[i]: f"{float(probs[0][i])*100:.2f}%" for i in range(len(labels))}

    # Add to log
    log_entry = f"Input: {text}\nPrediction: {pred_label} ({confidence*100:.2f}%)\nProbabilities: {all_probs}"
    prediction_log.append(log_entry)

    # Return predicted label + all probabilities + full log
    return pred_label, all_probs, "\n\n".join(prediction_log)

# --- Gradio Interface ---
interface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=4, placeholder="Enter your text here..."),
    outputs=[
        gr.Textbox(label="Predicted Label"),
        gr.JSON(label="Probabilities for All Labels"),
        gr.Textbox(label="Prediction Log", lines=10)
    ],
    title="Mental Health Text Classifier (RoBERTa)",
    description="Classifies text as Anxiety, Depression, Lonely, Suicide Watch, or Mental Health. Shows probabilities and keeps a log below.",
    theme="soft"
)

interface.launch()
