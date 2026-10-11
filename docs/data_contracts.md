# Data contracts

All inputs live under one workspace directory and are referenced from `lara_app.json` with paths relative to that directory. `lara-skills toy-data --output <dir>` writes a complete example of every table and a configuration that points at it.

## Application configuration (`lara_app.json`)

| Section | Keys | Notes |
| --- | --- | --- |
| top level | `version`, `market_id`, `seed` | `version` names the output folders (`data/processed/<version>/...`, `artifacts/models/<version>_<variant>`) |
| `encoder` | `backend` (`sentence_transformers` or `hashed_ngram`), `model_id`, `revision` (40 characters), `dimension`, `max_seq_length`, `batch_size`, `device`, `cpu_threads` | one frozen encoder for skills, occupations and descriptions. Default: BGE-M3 at the pinned revision used in the paper |
| `skills` | `path`, `sheet_name` (Excel), `id_column`, `name_column`, `taxonomy_columns`, `text_template`, `taxonomy_separator` | `text_template` receives `{skill_name}` and `{taxonomy_path}` |
| `postings` | `files`, `columns` (`source`, `posting_id`, `year`, `month`, `skills`, `occupation`), `skill_separator`, `occupation_code_width`, `start_year`, `end_year`, `threads`, `memory_limit`, `batch_size` | DuckDB settings for cleaning |
| `graph` | `thresholds` (`min_skill_postings`, `min_cooccurrence`, `weight_metric` cosine/jaccard, `min_weight`, `min_active_years`), `grid` | inclusive cutoffs; the grid feeds the threshold diagnostics table |
| `occupations` | `path`, `sheet_name`, `code_column`, `name_column`, `path_columns`, `code_width`, `embedding_template`, `min_occupation_skill_postings`, `threshold_grid` | `embedding_template` receives `{name}` and `{path}` |
| `patents` | `dir`, `files` (`patents`, `labels`, `candidate_pool`, `task_status`), `columns` (`application`, `year`, `text`, `family`), `split_fractions`, `split_seed` | the identifier column is always `sample_id` |
| `model` | `variant` (`lara` / `lara-na`), `device`, `cpu_threads`, hyperparameters (see [models.md](models.md)) | defaults are the paper settings |
| `evaluation` | `ks` | cutoffs for Recall@K and Hit@K |

Unspecified keys take the defaults listed in `lara_skills/app.py`.

## Skill dictionary

One row per skill. Required columns (names configurable): skill code, skill name, at least one taxonomy column. Codes must be unique strings; postings and judgments refer to skills by this code. Taxonomy columns form the hierarchy used by the model (for example main group > sub group > detail group); every level must be filled. CSV files are read as UTF-8 (with or without BOM), Excel needs the `excel` extra (`openpyxl`).

## Job postings

Parquet files, typically one per year. Required columns (names configurable):

| Column | Content |
| --- | --- |
| source | platform or data source identifier |
| posting id | identifier unique within the source; duplicates with identical content are merged, conflicting duplicates stop the run |
| year, month | integers; rows outside `[start_year, end_year]` or with an invalid month are excluded and counted |
| occupation | occupation code; left-padded with zeros to `occupation_code_width` |
| skills | skill codes joined by `skill_separator`; repeated codes are de-duplicated; unknown codes stop the run |

Postings without skills still count toward occupation posting totals. The cleaning stage writes standardized yearly files, a skill frequency table and a quality summary.

## Occupation table

One row per occupation: code (zero-padded to `code_width`), name, and taxonomy path columns (merged cells in Excel exports are forward-filled). Every occupation code in the postings must exist here. Occupations that never appear in postings are kept as nodes without relations.

## Technology descriptions (`patents.parquet`)

| Column | Content |
| --- | --- |
| `sample_id` | unique identifier used by all judgment tables |
| application id (`columns.application`) | descriptions sharing an application id are placed in the same split |
| year (`columns.year`) | integer; splits are balanced per year |
| text (`columns.text`) | the description that is encoded (title plus abstract works well) |
| family id (`columns.family`) | optional; descriptions sharing a family id are placed in the same split (empty string or null when unknown) |

Descriptions with identical normalized text are also grouped. Splitting uses only these columns, never the judgments.

## Relevance judgments

Three Parquet files in `patents.dir`:

| File | Rows | Required columns |
| --- | --- | --- |
| `labels.parquet` | numeric judgments only | `sample_id`, `perspective` (always `"skills"`), `skill_code`, `skill_name`, `relevance_score` in {0, 1, 2} |
| `candidate_pool.parquet` | every judged candidate, including withheld ones | same columns; `relevance_score` null for withheld judgments; `eligible_numeric_label` true/false |
| `task_status.parquet` | one row per description | `sample_id`, `perspective`, any status columns |

Grades: 2 = directly relevant, 1 = indirectly relevant, 0 = judged irrelevant. Skills that were never judged for a description are unknown, not negative: the training objective keeps soft target mass on them and the evaluation reports recall of known positives only. Every label must also appear in the candidate pool, keys must be unique, and skill codes must exist in the dictionary.

## Outputs

Each stage writes a versioned directory with a `manifest.json` (status, configuration, file hashes, package version, command line, module hashes) and refuses to overwrite an existing directory. Stage outputs are read back through their manifests, so edited files are detected. Training writes the checkpoint (`hgt_conditional.pt` or `hgt_network.pt`), validation metrics for the model and for the popularity, semantic and diffusion baselines, per-epoch history, top-100 lists and attention diagnostics. Evaluation writes `retrieval.csv`, per-query metrics, known-label ranks, top candidates and the full score matrix.
