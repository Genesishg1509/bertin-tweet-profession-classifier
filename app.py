"""Gradio demo: paste a tweet, get a profession-mention prediction.

Run locally with::

    python app.py

Meant to be deployed on Hugging Face Spaces (SDK: Gradio) so the model can
be tried in a browser without installing anything.
"""
from __future__ import annotations

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocess import clean_text

MODEL_DIR = "models/bertin"  # local path, or a HF Hub repo id once pushed there
THRESHOLD = 0.42  # from train.py's validation threshold tuning -- update after training

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def predict(tweet: str):
    if not tweet.strip():
        return {"SÍ menciona profesión": 0.0, "NO menciona profesión": 1.0}, ""

    text = clean_text(tweet)
    inputs = tokenizer(text, padding="max_length", truncation=True,
                       max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    prob_positive = torch.softmax(logits, dim=1)[0, 1].item()

    probabilities = {
        "SÍ menciona profesión": prob_positive,
        "NO menciona profesión": 1 - prob_positive,
    }
    decision = "SÍ" if prob_positive >= THRESHOLD else "NO"
    verdict = f"Predicción (umbral {THRESHOLD}): **{decision}** menciona profesión"

    return probabilities, verdict


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder="Pega un tweet en español...",
                      label="Tweet"),
    outputs=[gr.Label(label="Probabilidades"), gr.Markdown(label="Veredicto")],
    title="Detector de menciones a profesiones sanitarias",
    description=(
        "BERTIN RoBERTa afinado sobre ProfNER (tweets de la pandemia COVID-19). "
        "El umbral de decisión fue ajustado en el set de validación -- "
        "ver el README para la salvedad honesta sobre qué significa ese número."
    ),
    examples=[
        ["Las enfermeras y médicos están agotados tras meses de pandemia."],
        ["Hoy salí a caminar por el parque, hace un día precioso."],
    ],
)

if __name__ == "__main__":
    demo.launch()
