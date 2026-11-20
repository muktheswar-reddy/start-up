# MORPH-CG 04 - Path Sampling
#
# Goal: Sample α configurations and store exact recipes.
# Deliverable: Composition-path dataset in results/paths.json

import json
from pathlib import Path
import numpy as np

# Use the path_sampler module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from morph.path_sampler import sample_path, paths_to_dicts

# 1. Sample a composition path
# For two parents: alpha in [0, 1]
# alpha=0 -> pure parent A, alpha=1 -> pure parent B

print("Sampling composition path...")

paths = sample_path(
    n_parents=2,
    strategy="grid",
    resolution=10,  # 11 points: alpha in [0, 0.1, 0.2, ..., 1.0]
    seed=42,
)

# 2. Save the path dataset
path_data = paths_to_dicts(paths)
Path("results/paths.json").write_text(json.dumps(path_data, indent=2))

# 3. Also save as experiment manifest entry
manifest = {
    "experiment_id": "morph-cg-v0.1",
    "sampling": {
        "strategy": "grid",
        "resolution": 10,
        "seed": 42,
        "n_parents": 2,
    },
    "paths": path_data,
}

Path("results/experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

print(f"Sampled {len(paths)} configurations:")
for p in paths:
    print(f"  {p.index}: alpha={p.coefficients[1]:.2f}")

print("\nPaths saved to results/paths.json")
print("Manifest saved to results/experiment_manifest.json")