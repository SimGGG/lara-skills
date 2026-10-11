# Pipeline

`lara-skills build --config lara_app.json` runs the stages below in order. Each stage reads the manifests of its inputs, writes one versioned output directory and a run log under `artifacts/runs/<run_id>/`, and is skipped on rerun when its output manifest says `completed`. `--stage <name>` (repeatable) runs a subset in pipeline order.

| Stage | Module | Reads | Writes (under `data/processed/<version>/` unless noted) |
| --- | --- | --- | --- |
| `skill-embeddings` | `skill_embeddings.generate` | skill dictionary | `skill_embeddings/`: `skill_embeddings.npy` (unit vectors), `skill_index.parquet/.csv` (row alignment, taxonomy path, embedding text), `validation.json` |
| `clean-postings` | `postings.clean_postings` | posting files, skill index | `postings/`: yearly standardized postings, `skill_frequency.parquet`, `duplicate_groups.parquet`, `quality_summary.json` |
| `cooccurrence` | `postings.aggregate_cooccurrence` | cleaned postings | `cooccurrence/`: `nodes.parquet` (observed skills with yearly counts), `pairs.parquet` (co-occurrence counts, cosine, jaccard, active years), yearly pair files |
| `threshold-evaluation` | `graphs.evaluate_thresholds` | co-occurrence statistics | `artifacts/threshold_evaluation/<version>/threshold_metrics.csv`: structural diagnostics for every grid point (edges, isolates, components, degree) |
| `skill-graph` | `graphs.build_graph` | co-occurrence statistics, thresholds | `skill_graph/`: `nodes.parquet`, `edges.parquet` (undirected, once), `thresholds.json`, `metrics.json` |
| `candidates` | `training_inputs.prepare_candidates` | skill embeddings, skill graph | `skill_candidates/`: `candidates.parquet` (the complete dictionary with posting counts and graph status), `edges.parquet`, `neighbor_mean_weights.npz` (row-normalized adjacency), `feature_reference.json` |
| `splits` | `training_inputs.prepare_splits` | descriptions, judgments, candidates | `patent_splits/`: `assignments.parquet`, `train/`, `validation/`, `test/` with `patent_inputs`, `labels`, `judged_candidates`, `tasks` |
| `patent-embeddings` | `patent_embeddings.generate_patent_embeddings` | descriptions, split assignments, skill embedding manifest | `patent_embeddings/`: `query_embeddings.npy`, `query_index.parquet`, `encoder_config.json` |
| `hierarchy` | `hierarchy.prepare_hierarchy` | candidates, edges | `skill_hierarchy/`: taxonomy nodes per level, pooling matrices, coarse market graphs, membership |
| `occupations` | `occupations.prepare_occupations` | cleaned postings, occupation table, candidates | `occupation_graph/`: occupation nodes and vectors, occupation-skill counts and weights, row-normalized relations in both directions, `threshold_evaluation.csv` |
| `train` | `train_retrieval.fit_retrieval_recommenders` | candidates, hierarchy, occupations, query embeddings, train/validation splits | `artifacts/models/<version>_<variant>/`: checkpoint, validation metrics, baselines, history, diagnostics |

Afterwards:

- `lara-skills evaluate --split validation|test` (`evaluate.evaluate_split`) scores every candidate for the chosen split and writes `artifacts/evaluation/<version>_<variant>_<split>/`. The test split is loaded only through this explicit path; training code refuses it.
- `lara-skills recommend --text ...` (`recommender.RecommendationEngine`) encodes the description in memory, scores all candidates and prints the top list. The description is not stored.

## Rules that the stages enforce

- The complete dictionary is always the candidate set. Graph thresholds change relations, never which skills can be recommended. Skills without neighbors are scored from their own features.
- Co-occurrence is counted once per posting (binary incidence). Edge weights are cosine (default) or jaccard; cutoffs are inclusive.
- Occupation-skill weights are `n_os / sqrt(n_o * n_s)`, thresholded by minimum joint postings, then row-normalized separately for each direction.
- Splits are assigned from identifiers, years and text only, with year-balanced largest-remainder targets; application, family and exact-text groups never straddle splits.
- Training uses train labels, selects the checkpoint on validation known-positive Recall@5 (then Hit@5, then the earliest epoch, including epoch 0 which reproduces the semantic baseline), and never reads the test split.
- Every output is written to a temporary directory and renamed on success; failures leave no partial output and are recorded in the run log.

## Resuming and versioning

Outputs are never overwritten. To rerun a stage with different settings, change `version` in `lara_app.json` (new folders) or delete the stage's output directory. Variants share the prepared inputs and differ only in the model directory.
