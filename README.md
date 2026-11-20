# MORPH-CG: Compositional Integrity & Emergence Verification

A research prototype for **composition-path assurance** of combined AI models.

## What This Is

This is a research prototype (not a product) that implements a systematic method for detecting security-relevant behaviour that may emerge when AI models are merged/composed.

## Quick Start (Local)

```bash
pip install -r requirements.txt
python3 -m pytest tests/  # Run tests
python3 app.py           # Launch Gradio demo
```

## Hugging Face Space

This demo is deployed at [huggingface.co/spaces/muktheswarareddy/morph-cg](https://huggingface.co/spaces/muktheswar-reddy/morph-cg).

## Package Structure

```
morph-cg/
├─ morph/                    # Core package (8 modules)
│   ├─ config.py            # Experiment configuration
│   ├─ registry.py          # Model provenance tracking
│   ├─ composition.py       # Weight interpolation merge
│   ├─ path_sampler.py      # Composition path sampling
│   ├─ probes.py            # Test suite execution
│   ├─ metrics.py           # Behaviour comparison
│   ├─ attribution.py       # LOCO, Shapley analysis
│   └─ reporting.py         # Evidence bundles
├─ tests/                    # 58 pytest tests
├─ notebooks/                # 9-step Colab implementation
├─ research/                 # Documentation
└─ app.py                    # Gradio web demo
```

## Usage

```python
from morph import registry, composition, path_sampler

# Sample composition paths
paths = path_sampler.sample_path(n_parents=2, strategy="grid", resolution=10)

# Merge models
merged_weights = composition.linear_interpolation([weights_a, weights_b], [0.5, 0.5])

# Run probes
from morph.probes import run_probe_suite
results = run_probe_suite(suite, "model-a", predict_fn)
```

## Research Background

Based on the MORPH-CG Research and Startup Master Plan (18 Nov 2026). The hypotheses are that systematic composition-path sampling can detect security-relevant behaviours that component-only testing misses.

## License

MIT License

## Citation

If you use this code, please cite:
> MORPH-CG Research Prototype (2026) - Claude Code demonstration