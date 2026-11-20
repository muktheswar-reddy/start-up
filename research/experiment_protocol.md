# Experiment Protocol

## Pre-Registration Checklist

Before running the full experiment, the following must be defined and locked:

### 1. Models
- [ ] Select 2+ compatible encoder models (e.g., BERT-base, RoBERTa-base)
- [ ] Record model IDs, hashes, licenses, architectures
- [ ] Confirm public availability and commercial-use rights

### 2. Probes
- [ ] Define probe suite (text inputs, expected labels)
- [ ] Choose benchmark dataset (SST-2, AG News, or custom)
- [ ] Version the probe suite (hash the set)

### 3. Metrics
- [ ] Detection rate / recall definition
- [ ] False-positive rate definition
- [ ] Utility retention measurement
- [ ] Confidence interval method (bootstrap or analytical)

### 4. Sampling Strategy
- [ ] Grid resolution (e.g., 11 points for alpha in [0, 1])
- [ ] Random seed for reproducibility
- [ ] Number of random seeds for variance estimation

### 5. Baselines
- [ ] Component-only scanning: evaluate each parent, scan merged model
- [ ] Endpoint-only: test only final merge (alpha=0.5)
- [ ] Statistical anomaly: simple deviation detection

### 6. Success Thresholds (Pre-Registered)
- G1: Reproduce at least one published result
- G2: Path sampling improves detection by >10% absolute
- G5: Cost per assessment <$10 compute (on small models)

### 7. Failure Mode Analysis
- [ ] List scenarios where the method would be expected to fail
- [ ] Define what constitutes a "pivot" decision

## Execution Sequence

1. Load parent models → verify compatibility
2. Record parent baselines
3. Sample composition path → merge at each point
4. Run probes on all configurations
5. Compute metrics → compare across path
6. Run ablation analysis
7. Compare against baselines
8. Generate evidence bundle

## Data Management

- All model files stored in `data/models/` (or external Drive for Colab)
- All results stored in `results/`
- All provenance hashes stored in `results/manifest.json`
- All findings stored in `results/findings.csv`

## Reproducibility

- Git commit hash of experiment code included in manifest
- Random seeds logged
- Package versions logged in `environment.json`
- Exact merge coefficients recorded for each configuration
- Probe inputs and expected outputs versioned