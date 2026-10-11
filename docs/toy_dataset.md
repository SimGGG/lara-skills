# Toy dataset

`lara_skills.toy` generates a small synthetic labor market with a fixed seed. It exists to exercise every pipeline stage without downloading models or redistributing research data. Nothing in it is real, and results obtained on it say nothing about recommendation quality.

## Contents (`examples/toy/`, regenerated with `lara-skills toy-data --output <dir>`)

| Table | Rows | Construction |
| --- | --- | --- |
| `inputs/skills.csv` | 240 skills | taxonomy of 4 main groups x 3 sub groups x 2 detail groups; each detail group has 10 skills named `<detail group> <aspect>` (process setup, parameter tuning, fixture design, ...) |
| `inputs/occupations.csv` | 20 occupations | 4-digit codes, two-level taxonomy path; each occupation draws skills from three "home" detail groups with weights 0.6 / 0.25 / 0.15 |
| `inputs/postings/postings_<year>.parquet` | 24,432 rows for 2021-2024 | 24,000 postings with 3-8 skills each (12% get one random extra skill); occupations follow a Zipf-like distribution; plus 240 exact duplicates, 60 rows with month 13, 60 rows dated 2025 inside the 2024 file, 72 rows without posting id, and 2% rows with a repeated skill token |
| `inputs/patents/patents.parquet` | 60 descriptions | 12 per year 2019-2023; title plus abstract templates that mention the names of the core skills; two pairs share a family id |
| `inputs/patents/labels.parquet` | 512 numeric judgments | per description: 2-4 core skills (grade 2), 2-3 skills from the sibling detail group and 1 from a cross-domain group (grade 1), 1-3 skills from a distant group (grade 0) |
| `inputs/patents/candidate_pool.parquet` | 518 rows | the labels plus one withheld judgment (null grade) for every tenth description |
| `inputs/patents/task_status.parquet` | 60 rows | judged candidate counts per description |

Ten skills never appear in postings (unobserved) and twenty appear in only one or two postings (observed but isolated after thresholds), so the candidate set contains connected, isolated and unobserved skills.

## Encoder

The toy configuration uses the `hashed_ngram` backend: word unigrams, word bigrams and character trigrams hashed into 1024 signed buckets, log-count weighted, L2-normalized. It is deterministic across platforms and needs no model files. Descriptions that mention a skill's words therefore have a high cosine with that skill, which is enough to drive the pipeline but is not semantic similarity.

## What the quickstart produced on the reference machine

| Stage | Result |
| --- | --- |
| cleaning | 24,432 raw rows -> 24,000 postings (240 duplicates removed, 192 rows excluded for period or missing id); 230 observed skills |
| co-occurrence | 13,620 unique pairs; median support 2, 95th percentile 133 |
| skill graph (cosine >= 0.05, support >= 5) | 230 nodes, 3,431 edges, 20 isolates, largest component 91% of nodes |
| candidates | 240 skills: 210 connected, 20 isolated, 10 unobserved |
| splits | 58 groups; 35 / 15 / 10 descriptions in train / validation / test, exact year balance |
| occupation graph (>= 5 joint postings) | 20 occupations, 546 of 2,152 occupation-skill relations retained |
| training (`lara`, 30 epochs, CPU) | validation known-positive Recall@5 0.486 versus 0.463 for the semantic baseline; the whole quickstart ran in 17 seconds |

The exact numbers depend on the seed (default 20261011) and may differ slightly across hardware for the trained model; the prepared inputs are deterministic and their content hashes are recorded in `examples/toy/toy_manifest.json`, which the test suite and CI verify against the generator.
