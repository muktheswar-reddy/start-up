# MORPH-CG 01 - Literature and Benchmark
#
# Goal: Record benchmark, licence, threat model and baseline definitions.
# Deliverable: Experiment protocol and prior-art summary in research/

import json
from pathlib import Path

# 1. Load the literature matrix (you fill this as you read papers)
LITERATURE = [
    {
        "paper": "Wang et al. (USENIX Security 2025)",
        "title": "From Purity to Peril: Backdooring Merged Models",
        "model_family": "ViT, BERT, LLM",
        "threat": "MergeBackdoor",
        "method": "weight merging",
        "dataset": "SST-2, GLUE",
        "metrics": ["accuracy", "trigger_detection_rate"],
        "url": "https://www.usenix.org/conference/usenixsecurity25/presentation/wang-lijin",
    },
    {
        "paper": "Yuan et al. (ACL 2025)",
        "title": "Merge Hijacking",
        "model_family": "LLM",
        "threat": "backdoor during merging",
        "method": "LLM model merging",
        "dataset": "custom trigger benchmark",
        "metrics": ["trigger_success", "detection_rate"],
        "url": "https://aclanthology.org/2025.acl-long.1571/",
    },
    {
        "paper": "Microsoft LLM Backdoor Scanner (2026)",
        "title": "Detecting backdoored language models at scale",
        "model_family": "LLM",
        "threat": "backdoor scanning",
        "method": "motif extraction, trigger search",
        "dataset": "public model zoo",
        "metrics": ["precision", "recall", "FPR"],
        "repo": "https://github.com/microsoft/llm-backdoor-scanner",
    },
]

# 2. Save the literature matrix
Path("research/literature_review.md").parent.mkdir(exist_ok=True)
Path("data/literature_matrix.json").write_text(json.dumps(LITERATURE, indent=2))

# 3. Define the experiment protocol
PROTOCOL = {
    "hypothesis_H1": "Path sampling finds configurations missed by endpoint-only tests",
    "hypothesis_H2": "Interaction score improves detection over component scores",
    "hypothesis_H3": "Attribution method generalises across model families",
    "baseline_benchmarks": [
        "component_only_scanning",
        "endpoint_only_evaluation",
        "microsoft_llm_backdoor_scanner_applicable",
        "statistical_anomaly_detection",
    ],
    "evaluation_metrics": [
        "detection_rate",
        "false_positive_rate",
        "precision",
        "utility_retention",
        "compute_cost",
    ],
    "success_criteria": {
        "G1": "Reproduce at least one published result",
        "G2": "Method adds measurable value over baselines",
        "G5": "Runtime fits plausible review workflow",
    },
}

Path("data/experiment_protocol.json").write_text(json.dumps(PROTOCOL, indent=2))

print("Literature matrix saved to data/literature_matrix.json")
print("Experiment protocol saved to data/experiment_protocol.json")
print("\nKey papers to read:")
for l in LITERATURE:
    print(f"  - {l['paper']}: {l['url']}")