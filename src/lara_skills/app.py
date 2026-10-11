"""One application config -> per-stage configs, workspace layout and the build sequence.

Users describe their inputs once (skill dictionary, postings, occupation table,
patents with judgments) in ``lara_app.json``; every pipeline stage derives its
own configuration from that file and writes versioned outputs under the
workspace root. Completed stages are detected through their manifests and
skipped, so a build can be resumed.
"""

import copy
import json
from pathlib import Path

from . import graphs, hierarchy, occupations, patent_embeddings, postings, skill_embeddings, training_inputs
from .models import variant_model
from .train_retrieval import fit_retrieval_recommenders

STAGES = ["skill-embeddings", "clean-postings", "cooccurrence", "threshold-evaluation", "skill-graph",
          "candidates", "splits", "patent-embeddings", "hierarchy", "occupations", "train"]
ENCODER_DEFAULTS = {"backend": "sentence_transformers", "model_id": "BAAI/bge-m3",
                    "revision": "5617a9f61b028005a4858fdac845db406aefb181", "dimension": 1024,
                    "max_seq_length": 8192, "batch_size": 8, "device": "cpu", "cpu_threads": 8}
ENCODER_KEYS = list(ENCODER_DEFAULTS)
POSTINGS_DEFAULTS = {"skill_separator": ",", "occupation_code_width": None, "threads": 4,
                     "memory_limit": "4GB", "batch_size": 125000}
GRAPH_DEFAULTS = {"thresholds": {"min_skill_postings": 1, "min_cooccurrence": 5, "weight_metric": "cosine",
                                 "min_weight": 0.01, "min_active_years": 1},
                  "grid": {"min_skill_postings": [1, 5, 10], "min_cooccurrence": [1, 5, 10, 20, 50],
                           "min_weight": [0.0, 0.01, 0.025, 0.05, 0.1]}}
OCCUPATION_DEFAULTS = {"sheet_name": None, "code_width": 4, "embedding_template": "Occupation: {name}\nCategory: {path}",
                       "min_occupation_skill_postings": 5, "threshold_grid": [1, 2, 5, 10, 20]}
PATENT_DEFAULTS = {"files": dict(training_inputs.DEFAULT_GOLD_FILES),
                   "columns": {"application": "application_id", "year": "application_year",
                               "text": "technology_description", "family": "family_id"},
                   "split_fractions": {"train": 0.6, "validation": 0.2, "test": 0.2}, "split_seed": 42}
MODEL_DEFAULTS = {"variant": "lara", "device": "cpu", "cpu_threads": 8, "hidden_dim": 128, "decoder_dim": 64,
                  "layers": 2, "attention_heads": 4, "edge_weight_prior": 1.0, "node_attention_temperature": 0.2,
                  "node_attention_mode": "learned", "gate_dim": 32, "gate_temperature": 1.0, "dropout": 0.1,
                  "query_neighbors": 20, "query_temperature": 0.1, "epochs": 60, "learning_rate": 0.001,
                  "weight_decay": 0.0001, "gradient_clip": 5.0, "residual_bound": 0.1, "retrieval_temperature": 0.03,
                  "positive_target_mass": 0.5, "auxiliary_rank_weight": 0.05, "rank_temperature": 0.03,
                  "residual_regularization": 0.01, "checkpoint_scoring": True, "diffusion_alphas": [0.0, 0.25, 0.5, 0.75],
                  "retrieval_ks": [5, 10, 20, 50, 100], "score_chunk_size": 16384, "content_query_bound": 0.1,
                  "semantic_attention_temperature": 0.05, "occupation_route_temperature": 0.1,
                  "learned_attention_bound": 0.1, "routing_rank": 16, "routing_query_chunk": 8}
EVALUATION_DEFAULTS = {"ks": [5, 10, 20, 50, 100]}
REQUIRED = {"skills": ["path", "id_column", "name_column", "taxonomy_columns", "text_template", "taxonomy_separator"],
            "postings": ["files", "columns", "start_year", "end_year"],
            "occupations": ["path", "code_column", "name_column", "path_columns"],
            "patents": ["dir"]}


def _merge(defaults, values):
    result = copy.deepcopy(defaults)
    for key, value in (values or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_app_config(path):
    """Read ``lara_app.json``, fill documented defaults and check the required sections."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for section, keys in REQUIRED.items():
        if section not in raw:
            raise ValueError(f"Application config lacks the {section!r} section")
        missing = [k for k in keys if k not in raw[section]]
        if missing:
            raise ValueError(f"Section {section!r} lacks {missing}")
    app = {"version": raw.get("version", "v1"), "market_id": raw.get("market_id", "MARKET"), "seed": int(raw.get("seed", 42)),
           "encoder": _merge(ENCODER_DEFAULTS, raw.get("encoder")),
           "skills": dict(raw["skills"]),
           "postings": _merge(POSTINGS_DEFAULTS, raw["postings"]),
           "graph": _merge(GRAPH_DEFAULTS, raw.get("graph")),
           "occupations": _merge(OCCUPATION_DEFAULTS, raw["occupations"]),
           "patents": _merge(PATENT_DEFAULTS, raw["patents"]),
           "model": _merge(MODEL_DEFAULTS, raw.get("model")),
           "evaluation": _merge(EVALUATION_DEFAULTS, raw.get("evaluation")),
           "config_path": str(path)}
    for key in ["source", "posting_id", "year", "month", "skills", "occupation"]:
        if key not in app["postings"]["columns"]:
            raise ValueError(f"postings.columns lacks {key!r}")
    if app["patents"]["columns"].get("id", "sample_id") != "sample_id":
        raise ValueError("The patent identifier column must be named sample_id (labels and status tables use it)")
    app["patents"]["columns"]["id"] = "sample_id"
    variant_model(app["model"]["variant"])
    return app


def workspace_paths(app, variant):
    version = app["version"]
    base = f"data/processed/{version}"
    return {"skill_embeddings": f"{base}/skill_embeddings", "postings": f"{base}/postings",
            "cooccurrence": f"{base}/cooccurrence", "skill_graph": f"{base}/skill_graph",
            "skill_candidates": f"{base}/skill_candidates", "patent_splits": f"{base}/patent_splits",
            "patent_embeddings": f"{base}/patent_embeddings", "skill_hierarchy": f"{base}/skill_hierarchy",
            "occupation_graph": f"{base}/occupation_graph", "threshold_evaluation": f"artifacts/threshold_evaluation/{version}",
            "model": f"artifacts/models/{version}_{variant}", "evaluation": f"artifacts/evaluation/{version}_{variant}"}


def encoder_config(app, device=None):
    encoder = {key: app["encoder"][key] for key in ENCODER_KEYS}
    if device:
        encoder["device"] = device
    return encoder


def stage_config(app, stage, variant=None, device=None):
    """Compose the configuration one stage expects from the application config."""
    variant = variant or app["model"]["variant"]
    model_name = variant_model(variant)
    paths = workspace_paths(app, variant)
    encoder = encoder_config(app, device)
    skills, posting, graph, occupation, patents, model = (app[k] for k in ["skills", "postings", "graph", "occupations", "patents", "model"])
    thresholds = graph["thresholds"]
    if stage == "skill-embeddings":
        return {**encoder, "input_path": skills["path"], "sheet_name": skills.get("sheet_name"),
                "id_column": skills["id_column"], "name_column": skills["name_column"],
                "taxonomy_columns": list(skills["taxonomy_columns"]), "text_template": skills["text_template"],
                "taxonomy_separator": skills["taxonomy_separator"], "output_dir": paths["skill_embeddings"],
                "normalize_embeddings": True, "dtype": "float32", "seed": app["seed"]}
    if stage in {"clean-postings", "cooccurrence"}:
        return {"market_id": app["market_id"], "input_files": list(posting["files"]), "columns": dict(posting["columns"]),
                "skill_separator": posting["skill_separator"], "occupation_code_width": posting["occupation_code_width"],
                "start_year": int(posting["start_year"]), "end_year": int(posting["end_year"]),
                "skill_index": paths["skill_embeddings"] + "/skill_index.parquet",
                "clean_output": paths["postings"], "statistics_output": paths["cooccurrence"],
                "threads": posting["threads"], "memory_limit": posting["memory_limit"], "batch_size": posting["batch_size"]}
    if stage in {"threshold-evaluation", "skill-graph"}:
        return {"statistics_dir": paths["cooccurrence"], "evaluation_output": paths["threshold_evaluation"],
                "graph_output": paths["skill_graph"], "thresholds": dict(thresholds), "grid": dict(graph["grid"])}
    if stage in {"candidates", "splits"}:
        return {"embedding_dir": paths["skill_embeddings"], "graph_dir": paths["skill_graph"],
                "expected_graph_thresholds": {k: thresholds[k] for k in ["min_cooccurrence", "weight_metric", "min_weight"]},
                "candidate_output": paths["skill_candidates"], "gold_dir": patents["dir"], "gold_files": dict(patents["files"]),
                "split_output": paths["patent_splits"], "seed": int(patents["split_seed"]),
                "split_fractions": dict(patents["split_fractions"]), "patent_columns": dict(patents["columns"])}
    if stage == "patent-embeddings":
        return {**encoder, "skill_embedding_dir": paths["skill_embeddings"],
                "input_path": f"{patents['dir']}/{patents['files']['patents']}", "split_dir": paths["patent_splits"],
                "output_dir": paths["patent_embeddings"], "id_column": "sample_id", "text_column": patents["columns"]["text"],
                "task_mode": "single", "perspectives": ["skills"], "query_template": "{description}", "seed": app["seed"]}
    if stage == "hierarchy":
        return {"candidate_dir": paths["skill_candidates"], "taxonomy_columns": list(skills["taxonomy_columns"]),
                "output_dir": paths["skill_hierarchy"]}
    if stage == "occupations":
        return {"market_id": app["market_id"], "postings_dir": paths["postings"],
                "years": list(range(int(posting["start_year"]), int(posting["end_year"]) + 1)),
                "candidate_dir": paths["skill_candidates"], "dictionary_path": occupation["path"],
                "sheet_name": occupation.get("sheet_name"), "code_column": occupation["code_column"],
                "name_column": occupation["name_column"], "path_columns": list(occupation["path_columns"]),
                "code_width": int(occupation["code_width"]), "embedding_template": occupation["embedding_template"],
                "min_occupation_skill_postings": int(occupation["min_occupation_skill_postings"]),
                "threshold_grid": list(occupation["threshold_grid"]), "output_dir": paths["occupation_graph"],
                "threads": posting["threads"], "memory_limit": posting["memory_limit"], "encoder": encoder}
    if stage == "train":
        hyper = {k: v for k, v in model.items() if k not in {"variant", "device"}}
        return {**hyper, "candidate_dir": paths["skill_candidates"], "hierarchy_dir": paths["skill_hierarchy"],
                "occupation_dir": paths["occupation_graph"], "patent_embedding_dir": paths["patent_embeddings"],
                "split_dir": paths["patent_splits"], "output_dir": paths["model"],
                "device": device or model["device"], "seed": app["seed"], "task_mode": "single",
                "training_objective": "partial_label_retrieval", "score_mode": "bounded_residual", "top_k": 5,
                "training_models": [model_name], "default_engine": model_name, "variant": variant}
    if stage == "evaluate":
        return {"model_dir": paths["model"], "model": model_name, "device": device or model["device"],
                "ks": list(app["evaluation"]["ks"]), "variant": variant}
    raise ValueError(f"Unknown stage {stage!r}")


def output_dir(app, stage, variant=None):
    paths = workspace_paths(app, variant or app["model"]["variant"])
    return {"skill-embeddings": paths["skill_embeddings"], "clean-postings": paths["postings"],
            "cooccurrence": paths["cooccurrence"], "threshold-evaluation": paths["threshold_evaluation"],
            "skill-graph": paths["skill_graph"], "candidates": paths["skill_candidates"], "splits": paths["patent_splits"],
            "patent-embeddings": paths["patent_embeddings"], "hierarchy": paths["skill_hierarchy"],
            "occupations": paths["occupation_graph"], "train": paths["model"]}[stage]


def is_completed(root, relative):
    manifest = Path(root) / relative / "manifest.json"
    if not manifest.exists():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "completed"


def run_stage(root, app, stage, variant=None, device=None):
    """Run one stage unless its output already carries a completed manifest."""
    root = Path(root)
    target = output_dir(app, stage, variant)
    if is_completed(root, target):
        print(json.dumps({"skipped": stage, "output": target, "reason": "completed manifest exists"}), flush=True)
        return target
    config = stage_config(app, stage, variant, device)
    if stage == "skill-embeddings":
        skill_embeddings.generate(root, config)
    elif stage == "clean-postings":
        postings.clean_postings(root, config)
    elif stage == "cooccurrence":
        postings.aggregate_cooccurrence(root, config)
    elif stage == "threshold-evaluation":
        graphs.evaluate_thresholds(root, config)
    elif stage == "skill-graph":
        graphs.build_graph(root, config)
    elif stage == "candidates":
        training_inputs.prepare_candidates(root, config)
    elif stage == "splits":
        training_inputs.prepare_splits(root, config)
    elif stage == "patent-embeddings":
        patent_embeddings.generate_patent_embeddings(root, config)
    elif stage == "hierarchy":
        hierarchy.prepare_hierarchy(root, config)
    elif stage == "occupations":
        occupations.prepare_occupations(root, config)
    elif stage == "train":
        fit_retrieval_recommenders(root, config)
    else:
        raise ValueError(f"Unknown stage {stage!r}")
    return target


def build(root, app, variant=None, device=None, stages=None):
    """Run the requested stages in pipeline order; skip stages with completed outputs."""
    selected = [s for s in STAGES if stages is None or s in set(stages)]
    unknown = set(stages or []) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages {sorted(unknown)}; choose from {STAGES}")
    outputs = {}
    for stage in selected:
        outputs[stage] = run_stage(root, app, stage, variant, device)
    return outputs
