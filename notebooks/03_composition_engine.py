# MORPH-CG 03 - Composition Engine
#
# Goal: Implement a simple, validated weight interpolation and validate weights.
# Deliverable: Composed model checkpoints in results/merged.

import torch
import json
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Load two compatible parent models
MODEL_IDS = [
    "distilbert-base-uncased",  # Parent A: base model
    "distilbert-base-uncased-finetuned-sst-2-english",  # Parent B: fine-tuned
]

print("Loading parent models...")
tokens = [AutoTokenizer.from_pretrained(m) for m in MODEL_IDS]
models = [AutoModelForSequenceClassification.from_pretrained(m) for m in MODEL_IDS]
models = [m.eval() for m in models]

# 2. Simple linear interpolation between two weights
def linear_interpolate(model_a, model_b, alpha: float):
    """Interpolate between two model state dicts."""
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    merged_sd = {}
    for k in sd_a.keys():
        merged_sd[k] = (1 - alpha) * sd_a[k] + alpha * sd_b[k]
    merged = AutoModelForSequenceClassification.from_pretrained(MODEL_IDS[0])
    merged.load_state_dict(merged_sd)
    return merged.eval()

# 3. Create a merged configuration (alpha = 0.5 for now)
ALPHA = 0.5
print(f"Creating merged model with alpha={ALPHA}...")
merged_model = linear_interpolate(models[0], models[1], ALPHA)

# 4. Save the checkpoint
merged_path = Path(f"results/merged/alpha_{ALPHA:.2f}")
merged_path.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(merged_path)
tokens[0].save_pretrained(merged_path)

print(f"Merged model saved to {merged_path}")

# 5. Record merge recipe
recipe = {
    "method": "linear_interpolation",
    "parents": MODEL_IDS,
    "alpha": ALPHA,
    "parent_hashes": [torch.sum(p).item() for p in [m.state_dict() for m in models]],
}
Path("results/merged_recipe.json").write_text(json.dumps(recipe, indent=2))
print("Merge recipe saved to results/merged_recipe.json")