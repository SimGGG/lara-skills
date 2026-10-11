# Changelog

## 0.1.0 (2026-10-11)

First public release.

- LARA (`lara`) and LARA-NA (`lara-na`) rankers, typed graph attention, partial-label retrieval objective and validation-based selection, as used in the manuscript.
- Data pipeline from a skill dictionary, job postings, an occupation table and judged technology descriptions to a trained engine: skill features, posting cleaning, co-occurrence statistics, threshold diagnostics, skill graph, candidate set, grouped splits, description features, taxonomy hierarchy, occupation graph, training.
- `lara-skills` command line: `toy-data`, `build`, `evaluate`, `recommend`, `quickstart`.
- Frozen encoder backends: Sentence-Transformers (BGE-M3 default) and a dependency-free hashed n-gram encoder for the toy data and CI.
- Synthetic toy dataset with deliberate cleaning cases, isolated and unobserved skills, grouped splits and withheld judgments.
