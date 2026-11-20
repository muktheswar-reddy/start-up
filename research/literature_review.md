# Literature Review

## Core Reading List

1. **Wang et al. (USENIX Security 2025)** - "From Purity to Peril: Backdoering Merged Models From 'Harmless' Benign Components"
   https://www.usenix.org/conference/usenixsecurity25/presentation/wang-lijin
   - Reports MergeBackdoor: individually benign components can yield a backdoored merged model
   - Uses ViT, BERT, and LLM settings
   - **Implication**: Composition-level backdoors are real, not hypothetical

2. **Yuan et al. (ACL 2025)** - "Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models"
   https://aclanthology.org/2025.acl-long.1571/
   - Demonstrates backdoor attacks targeting LLM model merging
   - Shows inheritance of malicious behaviour into merged models
   - **Implication**: LLM merging is vulnerable to the same class of problems

3. **Microsoft (2026)** - "Detecting backdoored language models at scale"
   https://www.microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/
   - Provides LLM Backdoor Scanner repository: https://github.com/microsoft/llm-backdoor-scanner
   - Implements motif extraction and trigger reconstruction

4. **Li et al. (ICLR 2026)** - "Defending against Backdoor Attacks via Module Switching"
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbe25fa0e7c7084049276888a09acc8d-Abstract-Conference.html
   - Studies model-merging-based defence against backdoors
   - Shows that selective module combinations can mitigate attacks

## Standards

5. **ETSI TS 104 033 V1.1.1 (2026)** - Annex E: Model Bill of Materials
   - Defines provenance, consistency, integrity, and traceability requirements for AI computing platforms

6. **NIST AI RMF (2023)** - Generative AI Profile
   - Provides a risk-management framework for AI lifecycle governance

## Gap Analysis

| Work | Scope | Limitation | MORPH-CG Opportunity |
|------|-------|------------|---------------------|
| MergeBackdoor | Merge-induced backdoors | Component-level scanning | Composition-path coverage |
| LLM Backdoor Scanner | Trigger detection in LLMs | Binary classifier | Attribution & path analysis |
| Model Bill of Materials | Lineage tracking | No behavioural testing | Combined evidence bundle |

## Novelty Assessment

The literature confirms that model composition can create emergent security risks. MORPH-CG's **hypothesized novelty** lies not in discovering the problem, but in:

1. **Quantifying non-additive interactions** across a systematic sample of the composition space
2. **Attributing findings** to specific components or interactions via controlled ablation
3. **Producing reproducible evidence** that links behaviour to specific merge configurations

These are **testable hypotheses**, not validated claims. They become contributions only if experiments demonstrate superior detection, better attribution, or actionable insight over existing baselines.