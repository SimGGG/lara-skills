# lara-skills

**LARA** (Labor-market-Aware Relational Attention) reads a technology description, such as a patent abstract, and retrieves the relevant skills from a complete skill dictionary. Instead of matching text alone, it conditions the retrieval on a labor-market graph built from job postings: skills that co-occur in postings, occupations that demand them, and the dictionary's own taxonomy. A new description attends to those market nodes, routes occupation evidence to skills with description-specific weights, combines three relation routes through a learned gate, and applies a bounded correction to the frozen text similarity. Training uses partial human relevance judgments without treating unjudged skills as negatives.

This repository packages the architecture and the data pipeline so that the method can be rebuilt for another labor market from three kinds of inputs: a skill dictionary, job postings with occupation codes, and technology descriptions with relevance judgments. It ships a synthetic toy dataset that runs end to end on a laptop CPU in well under a minute. It does not ship the Korean research data or trained checkpoints (see [docs/reproducing_the_paper.md](docs/reproducing_the_paper.md)).

## Install

Python 3.11 or newer.

```bash
pip install git+https://github.com/SimGGG/lara-skills          # library and the lara-skills command
pip install "lara-skills[bge] @ git+https://github.com/SimGGG/lara-skills"   # adds sentence-transformers for BGE-M3
```

For development:

```bash
git clone https://github.com/SimGGG/lara-skills
cd lara-skills
pip install -e ".[dev]"
pytest
```

PyTorch is a dependency; a CPU build is enough for the toy data. Install a CUDA build of torch first if you want `--device cuda` for real-size inputs.

## Quickstart with the toy data

```bash
lara-skills quickstart --workspace lara_toy_workspace --device cpu
```

The command writes synthetic inputs into the workspace, runs every pipeline stage, trains LARA, evaluates it on the toy validation split and prints five recommendations for a sample description. The whole run takes about 20 seconds on a laptop CPU. Rerunning the command skips stages whose outputs are complete.

The toy data use a hashed n-gram text encoder so that nothing has to be downloaded. That encoder carries no learned semantics: the quickstart checks that the pipeline works, not how well the method recommends skills. Real deployments use a frozen Sentence-Transformers model (BGE-M3 by default).

Variants: `--variant lara` (default; the full model of the paper) or `--variant lara-na` (patent-conditioned node attention with fixed occupation routing). Both train from the same prepared inputs.

## Use your own labor market

1. Create a template configuration and inspect the toy inputs:

   ```bash
   lara-skills toy-data --output my_market        # writes inputs/ and lara_app.json
   ```

2. Replace the files under `my_market/inputs/` with your data and edit `my_market/lara_app.json` (paths, column names, thresholds, encoder, model settings). The required inputs are:

   | Input | Format | Content |
   | --- | --- | --- |
   | skill dictionary | CSV, Excel or Parquet | one row per skill: code, name, one or more taxonomy columns |
   | job postings | one Parquet file per year | source, posting id, year, month, occupation code, skill codes joined by a separator |
   | occupation table | CSV, Excel or Parquet | occupation code, name, taxonomy columns |
   | technology descriptions | Parquet | `sample_id`, application id, year, description text, family id |
   | relevance judgments | three Parquet files | numeric grades 0/1/2 per (`sample_id`, skill), the full judged pool including withheld judgments, and a per-description status table |

   Column names, separators and thresholds are configurable; see [docs/data_contracts.md](docs/data_contracts.md).

3. Build the engine, evaluate it and ask for recommendations:

   ```bash
   lara-skills build --config my_market/lara_app.json --device cuda
   lara-skills evaluate --config my_market/lara_app.json --split validation
   lara-skills recommend --config my_market/lara_app.json --text "Apparatus for laser welding of aluminum battery housings ..." --top-k 10
   ```

   `build` runs the stages in order (skill features, posting cleaning, co-occurrence, threshold diagnostics, skill graph, candidate set, grouped splits, description features, taxonomy hierarchy, occupation graph, training) and writes versioned outputs with manifests under the workspace. `evaluate --split test` is meant for a single final run after modeling choices are fixed on validation.

From Python:

```python
from lara_skills import RecommendationEngine

engine = RecommendationEngine("my_market", "artifacts/models/v1_lara", device="cpu")
engine.recommend_table("Apparatus for laser welding of aluminum battery housings ...", top_k=10)
```

## What is inside

| Area | Modules | Documentation |
| --- | --- | --- |
| data pipeline | `skill_embeddings`, `postings`, `graphs`, `training_inputs`, `patent_embeddings`, `hierarchy`, `occupations` | [docs/pipeline.md](docs/pipeline.md) |
| model | `attention_ranker` (typed graph attention), `network_attention_ranker` (LARA-NA), `conditional_routing_ranker` (LARA), `retrieval_objective` | [docs/models.md](docs/models.md) |
| training and evaluation | `train_retrieval`, `retrieval_evaluation`, `evaluate` | [docs/models.md](docs/models.md) |
| application | `app` (one config, resumable build), `cli`, `recommender`, `toy` | [docs/toy_dataset.md](docs/toy_dataset.md) |

Every stage writes a `manifest.json` with input hashes, the package version, the command line and module hashes, and refuses to overwrite an existing output; a run log is kept under `artifacts/runs/`. Model selection uses validation known-positive Recall@5 only.

## Relation to the paper

The code is the research implementation of *LARA: Labor-market-aware relational attention for technology-to-skill recommendation* (manuscript under review), reduced to the proposed architecture and its data pipeline. Benchmarks, ablation controls, the sealed final-test protocol and the figure scripts of the paper are not part of this package; [docs/reproducing_the_paper.md](docs/reproducing_the_paper.md) lists what the paper used and what this package can and cannot reproduce. [SOURCE_MAP.md](SOURCE_MAP.md) records how each module relates to the research code.

## Limitations

- Training and model selection assume a top-5 primary metric and bounded-residual scoring, as in the paper.
- Skills, occupations and descriptions must be encoded with the same frozen encoder; mixing encoders is rejected.
- Relevance judgments are partial: unjudged skills are never negatives, and reported recall covers known positives only.
- The method retrieves candidate skills in an observed labor-market context; it does not forecast adoption.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the software and the manuscript.

## License

MIT, see [LICENSE](LICENSE).
