# Novelty Hypotheses

## Core Hypotheses (H1 - H3 from the plan)

### H1: Path Sampling Finds Risky Configurations Missed by Endpoint-Only Tests

**Hypothesis**: A systematic composition-path sampling and interaction-attribution method can identify certain security-relevant behaviours that are missed by testing only the parent models and the final merged model.

**Evidence Required**:
- Matched endpoint-only baseline comparison
- Path-sampling ablation analysis
- Repeated trials across multiple random seeds
- Pre-registered detection threshold

### H2: Interaction Score Improves Detection Over Component Scores

**Hypothesis**: The interaction attribution method improves detection over simple component-only scanning and standard behavioural testing.

**Evidence Required**:
- Pre-registered model and features
- Held-out test set (not used in path sampling)
- AUROC/AUPRC and FPR at fixed recall
- Comparison with component scores and standard behavioural testing

### H3: Attribution is Useful, Not Merely Plausible

**Hypothesis**: The controlled ablation attribution identifies influential components/interactions with useful accuracy.

**Evidence Required**:
- Known component/interaction ground truth in controlled experiments
- Attribution precision/recall metrics
- Path replication tests (repeat along alternative paths)

## Additional Hypotheses

### H4: The Method Generalises

**Hypothesis**: The methodology generalises across at least two model families and two merge approaches.

**Evidence Required**:
- Test on at least 2 different model architectures (e.g., BERT and RoBERTa)
- Test on at least 2 merge methods (linear interpolation and task vector merging)
- Document performance drop as baseline for future work

### H5: The Added Assurance is Affordable

**Hypothesis**: Runtime and compute cost fit a plausible model review workflow.

**Evidence Required**:
- GPU time, CPU time, memory, and cost per model/configuration measured
- Cost per assessed model estimated
- Comparison with baseline workflow costs

## What NOT to Claim

- Do not claim "the first AI model security scanner"
- Do not claim "the first model provenance or lineage system"
- Do not claim "the first detector of backdoors in merged models"
- Do not claim that a low risk score means a model is safe
- Do not claim patentability based only on this report or keyword search

## Evaluation Gate

Proceed only if experiments show the method adds measurable value over baselines at acceptable false-positive rates and compute cost. The strongest contribution is a *measurable method*, not a promise of complete detection.