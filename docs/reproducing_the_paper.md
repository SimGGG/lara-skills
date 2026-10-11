# Reproducing the paper

This package contains the architecture and pipeline used for *LARA: Labor-market-aware relational attention for technology-to-skill recommendation*. It does not contain the data, and it deliberately leaves out the comparison models and the sealed evaluation protocol. This page lists what the paper used so that readers can judge what a rerun on their own data corresponds to.

## Data used in the paper (not redistributed)

| Input | Paper | Why it is not included |
| --- | --- | --- |
| job postings | 7.93 million cleaned Korean postings, 2021-2024, with 4-digit occupation codes and dictionary skill codes assigned by the data provider | licensed platform data |
| skill dictionary | Korean skill dictionary version 2.2, 6,558 skills with a three-level taxonomy | provided by the Korea Labor Institute under terms that do not permit redistribution |
| occupation table | Korean Employment Classification of Occupations 2018 (450 four-digit occupations) | distributed by its publisher |
| technology descriptions | 200 Korean patents (title plus abstract), 40 per application year 2019-2023, from a root-industry patent collection | collection terms |
| relevance judgments | human grades (2 direct, 1 indirect, 0 irrelevant, withheld) over a candidate pool of 1,932 pairs; two work-type perspectives merged into one task by union and maximum grade | contains the dictionary's skill names |

Trained checkpoints are bound to the candidate table, which lists the dictionary's skill names, so they are not distributed either.

## Settings used in the paper

- Encoder: `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181`, CLS pooling, float32, L2-normalized, no truncation (maximum 8,192 tokens). Skill text: name plus taxonomy path. Occupation text: name plus path. Description text: title plus abstract, encoded as is.
- Skill graph: co-occurrence counted once per posting; edges kept with support >= 5 and cosine >= 0.01; isolated and unobserved skills kept as candidates.
- Occupation relations: `n_os / sqrt(n_o * n_s)`, minimum 5 joint postings, row-normalized per direction.
- Splits: 120 / 40 / 40 descriptions, seed 42, grouped by application, family and identical text, balanced per year.
- Model and training: the defaults in `lara_skills/app.py` (hidden 128, decoder 64, 2 layers, 4 heads, 60 epochs, AdamW 1e-3, residual bound 0.1, retrieval temperature 0.03, positive target mass 0.5). Three training seeds (42, 43, 44); checkpoint selection on validation known-positive Recall@5.
- Test evaluation: one sealed pass over the 40 held-out descriptions with the fixed checkpoints.

Running `lara-skills build` with these defaults on comparable inputs reproduces the preparation and training path of the proposed model for one seed. Repeat `build --variant ...` with different `seed` values and `version` names for multi-seed results.

## Not included

- Benchmarks (Text MLP, GraphSAGE, GraphSAGE-Full, R-GCN, PinSage, HGT, taxonomy-gated models) and the ablation controls (removing occupation, co-occurrence or all posting relations, degree-preserving rewiring, fixed occupation routing).
- The sealed final-test protocol that evaluates many frozen checkpoints at once, the semantic-similarity-bin analysis, the training-size study, bootstrap supplements and the figure and table scripts.
- The converter that merges two work-type judgment sets into the single task; this package expects single-task judgments directly.

The research code that contains these parts is kept with the project; `SOURCE_MAP.md` lists the modules that were copied, adapted or left out.
