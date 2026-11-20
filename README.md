# MORPH-CG: Compositional Integrity & Emergence Verification

A research prototype for **composition-path assurance** of combined AI models.

## What This Is

This is a **research prototype**, not a product. The hypothesis: when models are merged, security-relevant behaviour can emerge that is invisible when testing each component in isolation. This prototype implements a systematic method for:

1. Sampling the composition configuration space (the "path")
2. Probing each configuration
3. Measuring non-additive behaviour
4. Attributing findings to components or interactions
5. Producing reproducible evidence reports

## Project Structure

```
morph-cg/
├─ morph/                    # Core package (8 modules)
│   ├─ __init__.py
│   ├─ config.py            # Dataclasses for experiment config
│   ├─ registry.py          # Model registry (hashes, provenance)
│   ├─ composition.py       # Weight interpolation, merge adapter
│   ├─ path_sampler.py      # Sample alpha configurations
│   ├─ probes.py            # Fixed probe suite
│   ├─ metrics.py           # Behaviour comparison
│   ├─ attribution.py       # LOCO, perturbation, Shapley
│   ├─ risk.py              # Evidence aggregation
│   └─ reporting.py         # Bundle, JSON, CSV output
├─ notebooks/                # 9-step Colab implementation
│   00_project_setup.ipynb
│   01_literature_and_benchmark.py
│   02_parent_baselines.py
│   03_composition_engine.py
│   04_path_sampling.py
│   05_probe_and_metrics.py
│   06_interaction_attribution.py
│   07_baselines_and_evaluation.py
│   08_report_generation.py
├─ research/                 # Documentation
│   ├─ literature_review.md
│   ├─ threat_model.md
│   ├─ novelty_hypotheses.md
│   └─ experiment_protocol.md
├─ configs/
│   └─ experiment_v0_1.yaml
├─ tests/                    # Pytest suite (58 tests)
│   test_registry.py
│   test_composition.py
│   test_path_sampler.py
│   test_probes.py
│   test_metrics.py
│   test_attribution.py
│   test_reporting.py
├─ results/                  # Generated artifacts
├─ data/                     # Model files, benchmark data
└─ reports/                  # Research memos, PDFs
```

## Installation

In Colab or locally:

```bash
pip install torch transformers datasets numpy pandas scipy scikit-learn matplotlib pytest safetensors
```

## Run in Colab

1. Open `notebooks/00_project_setup.ipynb`
2. Run each cell in sequence:
   - Notebook 00: Setup environment
   - Notebook 01: Literature and benchmark definitions
   - Notebook 02: Load and score parent models
   - Notebook 03: Create merged configurations
   - Notebook 04: Sample composition path
   - Notebook 05: Run probes, compute metrics
   - Notebook 06: Attribution analysis
   - Notebook 07: Baseline comparison
   - Notebook 08: Generate evidence bundle

## Usage

```python
from morph import registry, composition, path_sampler, probes, metrics, attribution, reporting

# Example: register a model
reg = registry.ModelRegistry()
reg.register("bert-base-uncased", architecture="bert")

# Example: sample composition path
paths = path_sampler.sample_path(n_parents=2, strategy="grid", resolution=5)

# Example: linear interpolation
merged = composition.linear_interpolation([state_dict_a, state_dict_b], [0.5, 0.5])
```

## Research Gate Rules

- **G1**: Reproduce at least one published result
- **G2**: Show >10% absolute improvement over endpoint-only testing
- **G5**: Keep compute cost under ~$10 for small-encoder prototype

## Disclaimer

> This is a **research prototype**. It demonstrates a method, not a validated product. Performance claims are based on controlled experiments with small models and benign synthetic probes. Do not use for production security decisions without independent validation.

## References

- Wang et al., "From Purity to Peril" (USENIX Security 2025)
- Yuan et al., "Merge Hijacking" (ACL 2025)
- Microsoft LLM Backdoor Scanner (2026)
- ETSI TS 104 033, NIST AI RMF