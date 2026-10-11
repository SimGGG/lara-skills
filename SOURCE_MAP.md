# Source map

Research package `candidate_skill_adoption` (private project, copied on 2026-10-11) -> `lara_skills`. 
Hashes are SHA-256 of the research source text at copy time; use them to detect drift when the research code changes.

## Copied or adapted modules

| Research module | Release module | Source SHA-256 | Change |
| --- | --- | --- | --- |
| `attention_ranker.py` | `attention_ranker.py` | `14449ede15f25ef5` | verbatim copy |
| `conditional_routing_ranker.py` | `conditional_routing_ranker.py` | `130db679e4d5e8e4` | verbatim copy |
| `embeddings.py` | `skill_embeddings.py` | `e497b90c02e86ae5` | rewritten around the encoder factory; accepts a config dict; CSV/Excel/Parquet dictionaries; tolerant package-version listing |
| `evaluation.py` | `evaluation.py` | `7d4615c5ab5ef9e4` | verbatim copy |
| `graph_encoder.py` | `graph_encoder.py` | `ec402b5ed7cdf538` | verbatim copy |
| `graphs.py` | `graphs.py` | `13d9ec45ca9ba172` | verbatim copy |
| `hierarchical_ranker.py` | `models.py` | `aebc881849b43d57` | only the ranker factory survives, restricted to hgt_conditional (lara) and hgt_network (lara-na); the hierarchy-gated models are not shipped |
| `hierarchy.py` | `hierarchy.py` | `7af4246b27b123cd` | script hash removed from the manifest |
| `network_attention_ranker.py` | `network_attention_ranker.py` | `d2a4f71134023a7d` | verbatim copy |
| `occupations.py` | `occupations.py` | `f7ee8b744ef0fe05` | CSV/Excel/Parquet dictionary input, encoder factory, feature dimension read from the array, script hash removed |
| `pipeline_io.py` | `pipeline_io.py` | `8a8be6cb5c0c313a` | rewritten: manifests record the installed package version, command line and module hashes instead of project-root script hashes; read_table helper added |
| `postings.py` | `postings.py` | `c589b0583ea9c214` | co-occurrence node table keeps every dictionary column instead of the fixed Korean taxonomy names |
| `ranking.py` | `ranking.py` | `160950d45ac1cf64` | verbatim copy |
| `recommender.py` | `recommender.py` | `d893960958fc641a` | two variants only; encoder factory; recommend_table added; model_dir is required |
| `recommender_data.py` | `recommender_data.py` | `e88107a2ea108120` | load_labeled_split added (explicit allow_test); load_development_split unchanged in behavior |
| `retrieval_evaluation.py` | `retrieval_evaluation.py` | `9dd3dd402a19905c` | retrieval_metrics kept; evaluate_saved_models removed |
| `retrieval_objective.py` | `retrieval_objective.py` | `4ceab2042cce5bb6` | verbatim copy |
| `text_encoder.py` | `encoders.py` | `464171fd41c43aef` | FrozenTextEncoder became SentenceTransformerEncoder; validate_vectors moved here; HashedNgramEncoder and make_encoder are new |
| `text_encoder.py` | `patent_embeddings.py` | `464171fd41c43aef` | encoder class moved to encoders.py, patent ID column renamed to sample_id, dimension read from the array, script hash removed |
| `train_retrieval.py` | `train_retrieval.py` | `8b9dec52bf637198` | two variants only; structure-control and benchmark branches removed; script hash removed; otherwise identical loop, loss and selection |
| `training_inputs.py` | `training_inputs.py` | `cc72bbb36f406f88` | judgment file names configurable (gold_files), judgment validation added, feature dimension read from the array, script hashes removed |

## New modules

| Release module | Purpose |
| --- | --- |
| `evaluate.py` | split evaluation of a saved engine (full-dictionary Recall@K / Hit@K; explicit test access) |
| `toy.py` | synthetic toy inputs and the toy application config |
| `app.py` | application config, per-stage config composition, resumable build |
| `cli.py` | command-line interface replacing the research scripts/*.py entry points |

## Research modules not shipped

`final_test.py`, `final_test_supplements.py`, `semantic_retrieval_study.py`, `supervision_retrieval_study.py`, `single_task_study.py`, `training_size_study.py`, `ranking_study.py`, `ranking_diagnostics.py`, `two_track.py`, `two_track_training.py`, `occupation_weights.py`, `structure_controls.py`, `benchmark_rankers.py`, `occupation_ranker.py`, `single_task.py`, `train_recommender.py`

These cover the paper's benchmarks, ablations, final test protocol, development history and figure/table generation. The public package ships the LARA architecture and the data pipeline needed to apply it.
