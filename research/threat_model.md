# Threat Model

## Protected Asset
The integrity and intended behaviour of a composed model **before deployment**.

## Attacker Capabilities

1. **Upstream Model Poisoning**: An upstream model or adapter may be crafted with hidden behaviour that only manifests post-composition.

2. **Merge Recipe Manipulation**: An attacker can influence which merge coefficients are used, targeting regions where new behaviour emerges.

3. **Probe Knowledge**: Attacker knows or can discover common probes or testing methodologies.

4. **Model Sourcing**: Models may come from untrusted sources (marketplaces, external collaborators, open-weight repositories).

## Defender Capabilities

1. **Parent Checkpoint Access**: The defender has access to parent model weights and can inspect them in isolation.

2. **Recipe Visibility**: The merge recipe and coefficient choices are known and auditable.

3. **Local Inference**: The defender can run inference locally on the parent and composed models.

4. **Controlled Experimentation**: The defender can create controlled modifications (ablations, coefficient variations) to test hypotheses.

## Scope & Out of Scope

### In Scope
- Emergent behaviour from weight-space merging of compatible models
- Non-additive interaction effects across merge coefficients
- Attribution of findings to component presence/absence
- Synthetic, controlled backdoors and benign anomalies

### Out of Scope (Initial Prototype)

| Category | Reason |
|----------|--------|
| Malicious model files | Running untrusted weights is a general security concern, not composition-specific |
| Corrupted training data | Data integrity is upstream of model training |
| Compromised GPU/host | Infrastructure security is separate from model composition |
| Data exfiltration via external tools | Not model-composition behaviour |
| Real-world malware, credential theft payloads | Safety scope: harmless synthetic behaviours only |
| API-only black-box models | Require different threat model (no weight access) |
| All prompt-trigger behaviours | Too broad; focus on merge-induced emergence |

## Attack Scenarios

### Scenario 1: Merge-Induced Backdoor
A benign-looking LoRA adapter has no trigger when tested in isolation. When merged with a base model at specific coefficients, the trigger activates. Component-only scanning misses this; composition-path testing detects it.

### Scenario 2: Emergent Capability
Two models trained for different tasks, when merged, exhibit a capability neither possessed. This may be useful (positive emergence) or harmful (negative emergence). The system must detect and attribute this.

### Scenario 3: Coefficient-Targeted Trigger
A hidden behaviour only activates at certain merge coefficients (e.g., α = 0.37 to 0.42). Testing only the endpoints (α=0, α=1) or midpoint (α=0.5) misses it. Path sampling aims to find such regions.

## Success Criteria

The method is successful if it:
1. Detects composition-specific cases missed by baseline methods
2. Provides actionable evidence for human review
3. Maintains acceptable utility (no severe performance degradation)
4. Runs within reasonable compute budgets