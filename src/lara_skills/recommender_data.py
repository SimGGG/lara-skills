"""Load versioned market features and labeled patent splits with integrity checks."""

import io
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .pipeline_io import digest
from .training_inputs import read_frame, verify_files


def load_market(root, candidate_dir):
    directory = Path(root) / candidate_dir
    verify_files(directory, ["candidates.parquet", "neighbor_mean_weights.npz", "feature_reference.json"])
    candidates = read_frame(directory / "candidates.parquet")
    if not np.array_equal(candidates.candidate_id, np.arange(len(candidates))):
        raise ValueError("Candidate rows are not aligned with IDs")
    reference = json.loads((directory / "feature_reference.json").read_text(encoding="utf-8"))
    path = Path(root) / reference["path_relative_to_project_root"]
    if digest(path) != reference["sha256"]:
        raise ValueError("Skill feature hash mismatch")
    features = np.load(io.BytesIO(path.read_bytes()), allow_pickle=False).astype(np.float32)
    adjacency = sparse.load_npz(io.BytesIO((directory / "neighbor_mean_weights.npz").read_bytes()))
    if features.shape[0] != len(candidates) or adjacency.shape != (len(candidates), len(candidates)):
        raise ValueError("Market feature/graph shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(adjacency.data).all():
        raise ValueError("Non-finite market input")
    freq = np.log1p(candidates.posting_count.to_numpy(dtype=np.float32))
    freq /= max(float(freq.max()), 1.)
    return candidates, features, adjacency, freq


def load_labeled_split(root, config, split, candidates, allow_test=False):
    """Queries, frozen query features and explicit judgments of one split.

    The test split is only returned when ``allow_test`` is set explicitly, so
    training and model selection code cannot read it by accident.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    if split == "test" and not allow_test:
        raise ValueError("Training/model selection loader does not accept the test split")
    root = Path(root)
    embedding_dir = root / config["patent_embedding_dir"]
    verify_files(embedding_dir, ["query_index.parquet", "query_embeddings.npy", "encoder_config.json"])
    encoder = json.loads((embedding_dir / "encoder_config.json").read_text(encoding="utf-8"))
    if config.get("task_mode", "work_types") != encoder.get("task_mode", "work_types"):
        raise ValueError("Training and encoder task modes differ")
    all_queries = read_frame(embedding_dir / "query_index.parquet")
    all_vectors = np.load(io.BytesIO((embedding_dir / "query_embeddings.npy").read_bytes()), allow_pickle=False)
    queries = all_queries[all_queries.split == split].copy().reset_index(drop=True)
    if config.get("task_mode") == "single" and (not queries.sample_id.is_unique
            or not queries.perspective.eq("skills").all() or not queries.perspective_id.eq(0).all()
            or encoder["perspectives"] != ["skills"] or encoder["query_template"] != "{description}"):
        raise ValueError("Single-task data must contain one plain query per patent")
    if queries.empty:
        raise ValueError(f"The {split} split has no queries")
    queries["query_id"] = np.arange(len(queries))
    features = all_vectors[queries.query_row.to_numpy()]
    split_dir = root / config["split_dir"]
    verify_files(split_dir, [f"{split}/labels.parquet", f"{split}/patent_inputs.parquet"])
    labels = read_frame(split_dir / split / "labels.parquet")
    labels = labels.merge(queries[["sample_id", "perspective", "query_id"]], on=["sample_id", "perspective"],
                          how="left", validate="many_to_one")
    labels = labels.merge(candidates[["skill_code", "candidate_id"]], on="skill_code", how="left", validate="many_to_one")
    if labels[["query_id", "candidate_id"]].isna().any().any() or not labels.relevance_score.isin([0, 1, 2]).all():
        raise ValueError("Missing candidate/query or invalid label")
    if labels.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("Duplicate judged candidate")
    # Derived pools use nullable integers; numeric labels must tensorize cleanly.
    labels["relevance_score"] = labels.relevance_score.astype(np.int64)
    labels = labels.sort_values(["query_id", "candidate_id"]).reset_index(drop=True)
    return {"queries": queries, "features": features, "labels": labels}


def load_development_split(root, config, split, candidates):
    if split not in {"train", "validation"}:
        raise ValueError("Training/model selection loader does not accept the test split")
    return load_labeled_split(root, config, split, candidates, allow_test=False)
