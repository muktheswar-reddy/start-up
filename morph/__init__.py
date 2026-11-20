"""MORPH-CG: Compositional Integrity & Emergence Verification.

A research prototype for composition-path assurance of composed AI models.

The package implements the modular architecture described in the MORPH-CG
research master plan.  Each module is independently testable and designed to
run on CPU first, with optional GPU support.

Modules
-------
registry        -- Model registry: record IDs, hashes, licences, revisions.
composition     -- Composition adapter: apply a supported merge method.
path_sampler    -- Sample the composition configuration space.
probes          -- Fixed, versioned probe suite.
metrics         -- Behavioural comparison and distribution distances.
attribution     -- Interaction scoring and ablation-based attribution.
reporting       -- Evidence reporter (JSON/CSV/manifest).
risk            -- Risk and uncertainty aggregation layer.
"""

__version__ = "0.1.0"