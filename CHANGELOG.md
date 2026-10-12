# Changelog

## Unreleased

- README: architecture figure, main results of the manuscript, shorter usage section.
- New `docs/results.md` with the full result tables and figures; method illustrations added to `docs/models.md`.
- Repository URL corrected to `SimGGG/lara-skills`.
- 2026-10-13: documentation aligned with the revised manuscript. Figures replaced by the revised exports (architecture figure with the decoder inputs, group vector from member skills, residual-norm bound in the attention figure, unified result figures with the "Middle" similarity bin); result pages add the decomposition of known positives by skill attribute, the training-size curves and the precision of the judged top-five lists; "seeds" renamed "training runs" in the result descriptions; data description updated (dictionary name, patent search, judgment procedure, second judgment round). No code change.

## 0.1.0 (2026-10-11)

First public release.

- LARA (`lara`) and LARA-NA (`lara-na`) rankers, typed graph attention, partial-label retrieval objective and validation-based selection, as used in the manuscript.
- Data pipeline from a skill dictionary, job postings, an occupation table and judged technology descriptions to a trained engine: skill features, posting cleaning, co-occurrence statistics, threshold diagnostics, skill graph, candidate set, grouped splits, description features, taxonomy hierarchy, occupation graph, training.
- `lara-skills` command line: `toy-data`, `build`, `evaluate`, `recommend`, `quickstart`.
- Frozen encoder backends: Sentence-Transformers (BGE-M3 default) and a dependency-free hashed n-gram encoder for the toy data and CI.
- Synthetic toy dataset with deliberate cleaning cases, isolated and unobserved skills, grouped splits and withheld judgments.
