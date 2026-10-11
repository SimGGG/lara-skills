"""Common skill candidates and label-independent grouped patent splits."""

import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .pipeline_io import digest, output_run, write_json, write_parquet

DEFAULT_GOLD_FILES = {"patents": "patents.parquet", "labels": "labels.parquet",
                      "candidate_pool": "candidate_pool.parquet", "task_status": "task_status.parquet"}


def read_frame(path):
    return pd.read_parquet(io.BytesIO(Path(path).read_bytes()))


def verify_files(directory, filenames):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Incomplete input: {directory}")
    for name in filenames:
        if digest(directory / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Input integrity check failed: {directory / name}")
    return manifest


def candidate_graph(index, graph_nodes, graph_edges):
    """Keep the complete dictionary; graph thresholds affect only relationships."""
    candidates = index.sort_values("embedding_row").reset_index(drop=True).copy()
    if not candidates.skill_code.is_unique or not np.array_equal(candidates.embedding_row, np.arange(len(candidates))):
        raise ValueError("Embedding index must have unique codes and contiguous rows")
    if not graph_nodes.skill_code.is_unique or not graph_nodes.node_id.is_unique:
        raise ValueError("Graph node keys must be unique")
    mapping = graph_nodes[["node_id", "skill_code", "posting_count"]].rename(columns={"node_id": "graph_node_id"})
    candidates = candidates.merge(mapping, on="skill_code", how="left", validate="one_to_one", sort=False)
    candidates.insert(0, "candidate_id", candidates.embedding_row.astype(np.int32))
    candidates["graph_node_id"] = candidates.graph_node_id.astype("Int64")
    candidates["posting_count"] = candidates.posting_count.fillna(0).astype(np.int64)
    candidates["observed_in_postings"] = candidates.posting_count > 0
    code_to_candidate = candidates.set_index("skill_code").candidate_id
    old_to_new = graph_nodes.set_index("node_id").skill_code.map(code_to_candidate)
    if old_to_new.isna().any():
        raise ValueError("Graph contains skills outside the common dictionary")
    edges = graph_edges.copy()
    for col in ["source_node", "target_node"]:
        values = edges[col].map(old_to_new)
        if values.isna().any():
            raise ValueError("Graph edge refers to an unknown node")
        edges[col] = values.astype(np.int32)
    src, dst = edges.source_node.to_numpy(), edges.target_node.to_numpy()
    weights = edges.weight.to_numpy(dtype=np.float32)
    if not np.isfinite(weights).all() or (weights <= 0).any() or (src == dst).any():
        raise ValueError("Expected finite positive edge weights without self-loops")
    if edges[["source_node", "target_node"]].duplicated().any():
        raise ValueError("Duplicate graph edges")
    adjacency = sparse.coo_matrix((np.r_[weights, weights], (np.r_[src, dst], np.r_[dst, src])),
                                  shape=(len(candidates), len(candidates))).tocsr()
    degree = np.diff(adjacency.indptr)
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1., row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    normalized = (sparse.diags(inverse) @ adjacency).tocsr()
    candidates["degree"] = degree
    candidates["is_isolated"] = degree == 0
    candidates["graph_status"] = np.where(~candidates.observed_in_postings, "unobserved",
                                           np.where(candidates.is_isolated, "isolated", "connected"))
    if not np.allclose(np.asarray(normalized.sum(axis=1)).ravel(), (degree > 0).astype(float), atol=1e-6):
        raise ValueError("Neighbor aggregation weights are not row-normalized")
    return candidates, edges, normalized


def prepare_candidates(root, cfg):
    root = Path(root)
    embedding = root / cfg["embedding_dir"]
    graph = root / cfg["graph_dir"]
    verify_files(embedding, ["skill_index.parquet", "skill_embeddings.npy"])
    verify_files(graph, ["nodes.parquet", "edges.parquet", "thresholds.json"])
    thresholds = json.loads((graph / "thresholds.json").read_text(encoding="utf-8"))
    for key, expected in cfg["expected_graph_thresholds"].items():
        if thresholds[key] != expected:
            raise ValueError(f"Graph threshold differs from experiment configuration: {key}")
    candidates, edges, adjacency = candidate_graph(read_frame(embedding / "skill_index.parquet"),
                                                   read_frame(graph / "nodes.parquet"),
                                                   read_frame(graph / "edges.parquet"))
    features = np.load(io.BytesIO((embedding / "skill_embeddings.npy").read_bytes()), allow_pickle=False)
    if features.ndim != 2 or features.shape[0] != len(candidates) or not np.isfinite(features).all():
        raise ValueError("Invalid candidate feature matrix")
    with output_run(root, root / cfg["candidate_output"], "common_skill_candidates", cfg) as (out, manifest):
        write_parquet(candidates, out / "candidates.parquet")
        candidates.to_csv(out / "candidates.csv", index=False, encoding="utf-8-sig")
        write_parquet(edges, out / "edges.parquet")
        buffer = io.BytesIO()
        sparse.save_npz(buffer, adjacency)
        (out / "neighbor_mean_weights.npz").write_bytes(buffer.getvalue())
        summary = {"candidates": len(candidates), "undirected_edges": len(edges),
                   "directed_nonzeros": adjacency.nnz, "feature_dimensions": features.shape[1],
                   "connected": int((candidates.degree > 0).sum()),
                   "observed_isolates": int((candidates.graph_status == "isolated").sum()),
                   "unobserved": int((candidates.graph_status == "unobserved").sum()),
                   "total_without_neighbors": int(candidates.is_isolated.sum()),
                   "gold_labels_used_for_candidates": False}
        write_json(out / "summary.json", summary)
        write_json(out / "feature_reference.json", {"path_relative_to_project_root": cfg["embedding_dir"] + "/skill_embeddings.npy",
                    "sha256": digest(embedding / "skill_embeddings.npy"), "row_key": "embedding_row",
                    "candidate_id_equals_embedding_row": True})
        write_json(out / "thresholds.json", thresholds)
        manifest.update({"summary": summary, "embedding_manifest_sha256": digest(embedding / "manifest.json"),
                         "graph_manifest_sha256": digest(graph / "manifest.json"),
                         "self_feature_policy": "always_use_self_features; zero_neighbor_aggregate_for_isolates"})
        print(json.dumps(summary), flush=True)


def group_patents(patents, columns):
    """Union same application, family or normalized exact text before splitting."""
    frame = patents.rename(columns={v: k for k, v in columns.items()}).copy()
    for col in ["id", "application", "year", "text"]:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing patent field: {col}")
    if not frame.id.is_unique:
        raise ValueError("Duplicate sample IDs")
    frame = frame.sort_values("id").reset_index(drop=True)
    frame["text_hash"] = frame.text.map(lambda t: hashlib.sha256(re.sub(r"\s+", " ", str(t).strip()).casefold().encode("utf-8")).hexdigest())
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for column in ["application", "family", "text_hash"]:
        seen = {}
        for i, value in enumerate(frame[column]):
            if pd.isna(value) or not str(value).strip():
                continue
            key = str(value).strip()
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    components = {}
    for i in range(len(frame)):
        components.setdefault(find(i), []).append(i)
    for indices in components.values():
        members = sorted(frame.loc[indices, "id"].tolist())
        group_id = "G-" + hashlib.sha256(json.dumps(members).encode()).hexdigest()[:16]
        frame.loc[indices, "group_id"] = group_id
    return frame


def assign_splits(grouped, fractions, seed):
    names = list(fractions)
    ratios = np.array(list(fractions.values()), dtype=float)
    if names != ["train", "validation", "test"] or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("Positive train/validation/test fractions must sum to one")
    years = sorted(grouped.year.unique())
    counts = pd.crosstab(grouped.group_id, grouped.year).reindex(columns=years, fill_value=0)
    # Largest-remainder allocation fixes year targets before looking at any labels.
    exact = ratios[:, None] * counts.sum(axis=0).to_numpy()[None, :]
    targets = np.floor(exact).astype(int)
    for j in range(len(years)):
        remaining = int(counts.iloc[:, j].sum() - targets[:, j].sum())
        order = np.argsort(-(exact[:, j] - targets[:, j]), kind="stable")
        targets[order[:remaining], j] += 1
    allocated = np.zeros_like(targets)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(counts.index.to_numpy()))
    ordered = sorted(shuffled, key=lambda g: -int(counts.loc[g].sum()))
    assignments = {}
    for group in ordered:
        vector = counts.loc[group].to_numpy()
        costs = []
        for split in range(len(names)):
            before = allocated[split] - targets[split]
            after = before + vector
            costs.append(float(((after ** 2 - before ** 2) / np.maximum(targets[split], 1)).sum()))
        chosen = int(np.argmin(costs))
        assignments[group] = names[chosen]
        allocated[chosen] += vector
    result = grouped.copy()
    result["split"] = result.group_id.map(assignments)
    if result.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("Group leaked across splits")
    return result, {"years": [int(y) for y in years], "split_names": names,
                    "target_patents_by_year": targets.tolist(), "actual_patents_by_year": allocated.tolist(),
                    "year_balance_exact": bool(np.array_equal(targets, allocated))}


def validate_judgments(sample_ids, labels, pool, tasks):
    """Fail early on malformed judgment tables; withheld grades belong in the pool only."""
    keys = ["sample_id", "perspective", "skill_code"]
    for frame, name in [(labels, "labels"), (pool, "candidate_pool")]:
        missing = set(keys + ["skill_name", "relevance_score"]) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {sorted(missing)}")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"Missing keys or duplicate judgments in {name}")
        if not frame.sample_id.isin(sample_ids).all():
            raise ValueError(f"{name} refers to patents outside the patent table")
    if labels.relevance_score.isna().any() or not labels.relevance_score.isin([0, 1, 2]).all():
        raise ValueError("labels must hold numeric grades 0, 1 or 2 only; keep withheld judgments in candidate_pool")
    if not labels.set_index(keys).index.isin(pool.set_index(keys).index).all():
        raise ValueError("Every label must also appear in candidate_pool")
    if not {"sample_id", "perspective"} <= set(tasks.columns) or tasks.duplicated(["sample_id", "perspective"]).any():
        raise ValueError("task_status needs unique (sample_id, perspective) rows")


def prepare_splits(root, cfg):
    root = Path(root)
    gold = root / cfg["gold_dir"]
    files = {**DEFAULT_GOLD_FILES, **cfg.get("gold_files", {})}
    # Splitting uses patent inputs alone, never relevance grades or graph coverage.
    inputs = read_frame(gold / files["patents"])
    grouped = group_patents(inputs, cfg["patent_columns"])
    assigned, allocation = assign_splits(grouped, cfg["split_fractions"], cfg["seed"])
    assignment = assigned[["id", "application", "year", "family", "text_hash", "group_id", "split"]].rename(
        columns={"id": "sample_id", "application": "application_id", "year": "application_year", "family": "family_id"})
    if set(assignment.split) != {"train", "validation", "test"}:
        raise ValueError("Not all three splits have samples")
    for col in ["application_id", "family_id", "text_hash", "group_id"]:
        valid = assignment[col].notna() & assignment[col].astype(str).str.strip().ne("")
        if (assignment[valid].groupby(col).split.nunique() > 1).any():
            raise ValueError(f"Split leakage in {col}")
    labels = read_frame(gold / files["labels"])
    pool = read_frame(gold / files["candidate_pool"])
    tasks = read_frame(gold / files["task_status"])
    validate_judgments(inputs[cfg["patent_columns"]["id"]], labels, pool, tasks)
    candidates_dir = root / cfg["candidate_output"]
    verify_files(candidates_dir, ["candidates.parquet"])
    candidates = read_frame(candidates_dir / "candidates.parquet")
    for frame in [labels, pool]:
        if not frame.skill_code.isin(candidates.skill_code).all():
            raise ValueError("Labeled skill missing from common candidate universe")
    with output_run(root, root / cfg["split_output"], "patent_group_split", cfg) as (out, manifest):
        write_parquet(assignment, out / "assignments.parquet")
        assignment.to_csv(out / "assignments.csv", index=False, encoding="utf-8-sig")
        totals = {}
        mapping = assignment[["sample_id", "split"]]
        for name, frame in [("patent_inputs", inputs), ("labels", labels), ("judged_candidates", pool), ("tasks", tasks)]:
            joined = frame.merge(mapping, on="sample_id", how="left", validate="many_to_one")
            if joined.split.isna().any():
                raise ValueError(f"Unassigned rows in {name}")
            for split in cfg["split_fractions"]:
                folder = out / split
                folder.mkdir(exist_ok=True)
                subset = joined[joined.split == split].copy()
                write_parquet(subset, folder / f"{name}.parquet")
                totals.setdefault(split, {})[name] = len(subset)
        summary = {"patents": len(inputs), "groups": assignment.group_id.nunique(), "seed": cfg["seed"],
                   "allocation": allocation, "split_rows": totals,
                   "application_family_exact_text_group_overlap": 0,
                   "missing_candidate_codes": 0, "gold_relevance_rows_preserved": len(labels),
                   "held_or_unjudged_labels_imputed": False, "relevance_used_to_assign_splits": False,
                   "approximate_semantic_duplicates_checked": False,
                   "graph_thresholds_tuned_with_test_labels": False}
        write_json(out / "summary.json", summary)
        manifest.update({"summary": summary,
                         "inputs": {name: digest(gold / name) for name in files.values()},
                         "candidate_manifest_sha256": digest(candidates_dir / "manifest.json"),
                         "split_inputs": "patent identifiers, years, family IDs and description text; relevance grades were not read for assignment"})
        print(json.dumps(summary), flush=True)
