# MORPH-CG 02 - Parent Baselines
#
# Goal: Load compatible parent models and evaluate fixed probes.
# Deliverable: Parent scorecard in results/parent_baselines.json

import torch
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Choose small, public, licensed research models
PARENT_MODELS = [
    {
        "id": "distilbert-base-uncased-finetuned-sst-2-english",
        "source": "huggingface",
        "task": "sentiment",
    },
    # Add a second compatible encoder for merging
    # "id": "prajjwal1/bert-mini",  # alternative
]

# 2. Simple probe suite (SST-2 style)
PROBES = [
    {"text": "This movie was great!", "label": 1},
    {"text": "Terrible film, waste of time.", "label": 0},
    {"text": "I enjoyed every minute.", "label": 1},
    {"text": "It was boring and slow.", "label": 0},
    {"text": "Amazing acting and plot.", "label": 1},
]

def load_model(model_id: str):
    """Load a model and tokenizer, return them."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

def run_probe(model, tokenizer, text: str):
    """Run a single probe, return logits and label."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()
    return pred, logits.squeeze().tolist()

# 3. Score each parent
results = {}
for m in PARENT_MODELS:
    mp_id = m["id"]
    print(f"Loading {mp_id}...")
    try:
        tokenizer, model = load_model(mp_id)
        model_hash = model.get_extension_state_extras().get("model_hash", "unknown")
    except Exception as e:
        print(f"  Error loading: {e}")
        continue

    parent_results = []
    for p in PROBES:
        pred, logits = run_probe(model, tokenizer, p["text"])
        correct = int(pred == p["label"])
        parent_results.append({
            "text": p["text"],
            "expected": p["label"],
            "predicted": pred,
            "correct": correct,
            "logits": logits,
        })

    results[mp_id] = {
        "task": m["task"],
        "accuracy": sum(r["correct"] for r in parent_results) / len(parent_results),
        "probes": parent_results,
    }
    print(f"  Accuracy: {results[mp_id]['accuracy']:.4f}")

# 4. Save the scorecard
Path("results").mkdir(exist_ok=True)
Path("results/parent_baselines.json").write_text(json.dumps(results, indent=2))
print("\nParent scorecard saved to results/parent_baselines.json")