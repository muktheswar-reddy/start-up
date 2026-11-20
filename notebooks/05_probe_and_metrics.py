# MORPH-CG 05 - Probe and Metrics
#
# Goal: Run probes; compute task-specific behaviour differences.
# Deliverable: Raw results + metric tables in results/metrics.json

import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from morph.probes import ProbeSuite, ProbeSpec, run_probe_suite
from morph.metrics import MetricResult, label_agreement_rate, label_flip_rate

# 1. Define the probe suite
PROBES = [
    ProbeSpec(
        name=f"probe_{i}",
        version=1,
        inputs=[text],
        expected_label=expected,
    )
    for i, (text, expected) in enumerate([
        ("This movie was great!", 1),
        ("Terrible film, waste of time.", 0),
        ("I loved this!", 1),
        ("Boring and awful.", 0),
        ("Fantastic acting.", 1),
        ("Disappointing story.", 0),
        ("Amazing plot twist.", 1),
        ("Slow and confusing.", 0),
    ])
]

suite = ProbeSuite(PROBES)

# 2. Load models for all configurations
parent_ids = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-finetuned-sst-2-english",
]

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

# 3. Run probes on each configuration
results = {}
for name, model_id in [("parents", parent_ids), ("all_configs", ["merged-alpha-0.5"])]:
    results[name] = {}

print("Running probe suite on parent models...")

for parent_id in parent_ids:
    tokenizer, model = load_model(parent_id)
    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        return int(torch.argmax(logits, dim=-1).item()), logits.squeeze().tolist()

    probes_results = run_probe_suite(suite, parent_id, predict)
    outputs = [r.output for r in probes_results]
    expected = [r.metadata.get("expected_label") for r in probes_results]

    metrics = {
        "agreement": label_agreement_rate(outputs, expected),
        "flip_rate": label_flip_rate(outputs, expected),
    }

    results["parents"][parent_id] = {
        "metrics": {k: m.to_dict() for k, m in metrics.items()},
        "n_correct": sum(1 for o, e in zip(outputs, expected) if o == e),
        "n_total": len(outputs),
    }
    print(f"  {parent_id}: accuracy={results['parents'][parent_id]['n_correct']/results['parents'][parent_id]['n_total']:.2%}")

# 4. Save results
Path("results/metrics.json").write_text(json.dumps(results, indent=2))
print("\nMetrics saved to results/metrics.json")

# Print summary table
print("\n=== Metric Summary ===")
for parent_id, data in results["parents"].items():
    m = data["metrics"]["agreement"]
    print(f"{parent_id}: {m['value']:.3f} (95% CI: [{m['ci_low']:.3f}, {m['ci_high']:.3f}])")