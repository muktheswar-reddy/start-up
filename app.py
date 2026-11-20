"""MORPH-CG Hugging Face Space app.

A Gradio interface for composition-path assurance of merged models.
https://huggingface.co/spaces/muktheswarareddy/morph-cg
"""

import gradio as gr
import json
from pathlib import Path

# Import MORPH-CG modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from morph.config import MergeRecipe, ProbeSpec
from morph.path_sampler import sample_path, paths_to_dicts
from morph.probes import ProbeSuite

def run_path_sampling_demo(n_parents: int = 2, resolution: int = 5, seed: int = 42):
    """Demonstrate composition-path sampling."""
    paths = sample_path(
        n_parents=n_parents,
        strategy="grid",
        resolution=resolution,
        seed=seed,
    )

    results = {
        "n_paths": len(paths),
        "paths": [
            {
                "index": p.index,
                "coefficients": [round(c, 3) for c in p.coefficients],
            }
            for p in paths[:10]  # Show first 10
        ],
        "strategy": "grid",
        "resolution": resolution,
    }
    return json.dumps(results, indent=2)

def generate_sample_paths():
    """Generate paths.json content."""
    paths = sample_path(n_parents=2, strategy="grid", resolution=10, seed=42)
    return paths_to_dicts(paths)

def show_package_info():
    """Show package information."""
    return """
**MORPH-CG v0.1.0**

Composition-Path Assurance Research Prototype

**Package modules:**
- `morph.config` - Experiment configuration dataclasses
- `morph.registry` - Model registry with SHA-256 provenance
- `morph.composition` - Linear interpolation merge adapter
- `morph.path_sampler` - Composition-path sampling (grid/random/boundary)
- `morph.probes` - Versioned test suite execution
- `morph.metrics` - Agrement rate, flip rate, JS divergence
- `morph.attribution` - LOCO, Shapley decomposition
- `morph.reporting` - Evidence bundle generation

**Research Goal:**
Detect security-relevant behaviour that emerges when AI models are merged/composed.
    """

def list_models():
    """List example models."""
    return """
**Compatible encoder models (examples):**
- distilbert-base-uncased
- distilbert-base-uncased-finetuned-sst-2-english
- bert-base-uncased
- bert-base-uncased-finetuned-sst-2-english
- roberta-base

**Usage:**
```python
from morph import registry, composition, path_sampler

# Register models
reg = registry.ModelRegistry()
reg.register("bert-base-uncased", architecture="bert")

# Sample paths
paths = path_sampler.sample_path(n_parents=2, strategy="grid")

# Merge
merged_weights = composition.linear_interpolation([sd_a, sd_b], [0.5, 0.5])
```
    """

# Gradio Interface
with gr.Blocks(title="MORPH-CG: Compositional Integrity & Emergence Verification") as demo:
    gr.Markdown("# MORPH-CG")
    gr.Markdown("**Compositional Integrity & Emergence Verification**")
    gr.Markdown("Research prototype for detecting security-relevant behaviour in merged AI models")

    with gr.Tabs():
        with gr.TabItem("Path Sampling Demo"):
            with gr.Row():
                n_parents = gr.Number(value=2, label="Number of Parents", minimum=2, maximum=5)
                resolution = gr.Slider(3, 20, value=10, step=1, label="Grid Resolution")
                seed = gr.Number(value=42, label="Random Seed")
            sample_btn = gr.Button("Sample Composition Path")
            output_json = gr.Textbox(label="Results", lines=15)
            sample_btn.click(fn=run_path_sampling_demo, inputs=[n_parents, resolution, seed], outputs=output_json)

        with gr.TabItem("Generate Paths JSON"):
            gen_btn = gr.Button("Generate paths.json")
            paths_output = gr.Code(label="paths.json Content", language="json")
            gen_btn.click(fn=generate_sample_paths, outputs=paths_output)

        with gr.TabItem("About"):
            about = gr.Markdown(show_model_info())

        with gr.TabItem("Models"):
            models = gr.Markdown(list_models())

        with gr.TabItem("Package Info"):
            pkg_info = gr.Markdown(show_package_info())

if __name__ == "__main__":
    demo.launch()