"""Label-independent taxonomy paths, mean pooling and market coarse graphs."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .pipeline_io import digest, output_run, write_json, write_parquet
from .training_inputs import read_frame, verify_files


def row_normalize(matrix):
    matrix = matrix.tocsr().astype(np.float32)
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1., totals, out=np.zeros_like(totals), where=totals > 0)
    return (sparse.diags(inverse) @ matrix).tocsr()


def build_hierarchy(candidates, edges, columns):
    """Use complete paths as IDs; no labels and no fabricated leaf edges."""
    n = len(candidates)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Taxonomy columns must be a nonempty ordered unique list")
    if not np.array_equal(candidates.candidate_id, np.arange(n)) or not candidates.skill_code.is_unique:
        raise ValueError("Candidate IDs must match feature rows")
    paths = candidates[columns].copy()
    if paths.isna().any().any():
        raise ValueError("Missing taxonomy paths; provide a complete market adapter")
    paths = paths.apply(lambda col: col.astype(str).str.strip())
    if paths.eq("").any().any():
        raise ValueError("Blank taxonomy paths")
    src, dst = edges.source_node.to_numpy(), edges.target_node.to_numpy()
    weight = edges.weight.to_numpy(dtype=np.float64)
    if (not np.isfinite(weight).all() or (weight <= 0).any() or (src == dst).any()
            or ((src < 0) | (src >= n) | (dst < 0) | (dst >= n)).any()):
        raise ValueError("Invalid market edges")
    unordered = np.sort(np.column_stack([src, dst]), axis=1)
    if len(np.unique(unordered, axis=0)) != len(edges):
        raise ValueError("Expected each undirected edge once")
    market = sparse.coo_matrix((np.r_[weight, weight], (np.r_[src, dst], np.r_[dst, src])), shape=(n, n)).tocsr()
    pools, coarse, memberships, nodes, stats = [], [], [], [], []
    previous = None
    for level in range(len(columns)):
        tuples = list(paths[columns[:level + 1]].itertuples(index=False, name=None))
        unique = sorted(set(tuples))
        ids = {path: i for i, path in enumerate(unique)}
        membership = np.array([ids[path] for path in tuples], dtype=np.int64)
        counts = np.bincount(membership, minlength=len(unique)).astype(np.float64)
        indicator = sparse.coo_matrix((np.ones(n), (np.arange(n), membership)), shape=(n, len(unique))).tocsr()
        pool = (sparse.diags(1. / counts) @ indicator.T).tocsr()
        # Average edge weight per possible cross-group skill pair, then normalize.
        group_market = (pool @ market @ pool.T).tocsr()
        group_market = (group_market - sparse.diags(group_market.diagonal())).tocsr()
        group_market.eliminate_zeros()
        normalized = row_normalize(group_market)
        for path, node_id in ids.items():
            nodes.append({"level": level, "level_name": columns[level], "node_id": node_id,
                          "path": json.dumps(path, ensure_ascii=False), "label": path[-1],
                          "parent_node_id": previous[path[:-1]] if previous is not None else None,
                          "descendant_skills": int(counts[node_id])})
        stats.append({"level": level, "column": columns[level], "nodes": len(unique),
                      "undirected_market_edges": int(normalized.nnz // 2),
                      "market_isolates": int((np.diff(normalized.indptr) == 0).sum())})
        pools.append(pool.astype(np.float32))
        coarse.append(normalized)
        memberships.append(membership)
        previous = ids
    return {"nodes": pd.DataFrame(nodes), "membership": np.column_stack(memberships),
            "pools": pools, "coarse": coarse, "levels": stats,
            "local_market_available": np.diff(market.indptr) > 0}


def prepare_hierarchy(root, config):
    root = Path(root)
    directory = root / config["candidate_dir"]
    verify_files(directory, ["candidates.parquet", "edges.parquet"])
    candidates = read_frame(directory / "candidates.parquet")
    bundle = build_hierarchy(candidates, read_frame(directory / "edges.parquet"), config["taxonomy_columns"])
    with output_run(root, root / config["output_dir"], "taxonomy_market_hierarchy", config) as (out, manifest):
        write_parquet(bundle["nodes"], out / "taxonomy_nodes.parquet")
        for key in ["membership", "local_market_available"]:
            buffer = io.BytesIO()
            np.save(buffer, bundle[key], allow_pickle=False)
            (out / f"{key}.npy").write_bytes(buffer.getvalue())
        for level, (pool, coarse) in enumerate(zip(bundle["pools"], bundle["coarse"])):
            for name, matrix in [("pool", pool), ("market", coarse)]:
                buffer = io.BytesIO()
                sparse.save_npz(buffer, matrix)
                (out / f"{name}_{level}.npz").write_bytes(buffer.getvalue())
        summary = {"candidate_count": len(candidates), "levels": bundle["levels"],
                   "taxonomy_nodes": len(bundle["nodes"]), "missing_paths": 0,
                   "leaf_market_isolates": int((~bundle["local_market_available"]).sum()),
                   "pooling": "equal_weight_mean_over_descendant_skills",
                   "coarse_market": "mean_edge_weight_per_skill_pair; remove_diagonal; row_normalize",
                   "gold_labels_used": False}
        write_json(out / "summary.json", summary)
        manifest.update({"summary": summary, "candidate_manifest_sha256": digest(directory / "manifest.json")})
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def load_hierarchy(root, hierarchy_dir, candidate_dir):
    root, directory = Path(root), Path(root) / hierarchy_dir
    manifest = verify_files(directory, ["summary.json"])
    if digest(root / candidate_dir / "manifest.json") != manifest["candidate_manifest_sha256"]:
        raise ValueError("Hierarchy and candidate market versions do not match")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    levels = len(summary["levels"])
    names = ["membership.npy", "local_market_available.npy"]
    names += [f"{name}_{i}.npz" for name in ["pool", "market"] for i in range(levels)]
    verify_files(directory, names)
    arrays = {key: np.load(io.BytesIO((directory / f"{key}.npy").read_bytes()), allow_pickle=False)
              for key in ["membership", "local_market_available"]}
    arrays["pools"] = [sparse.load_npz(io.BytesIO((directory / f"pool_{i}.npz").read_bytes())) for i in range(levels)]
    arrays["coarse"] = [sparse.load_npz(io.BytesIO((directory / f"market_{i}.npz").read_bytes())) for i in range(levels)]
    arrays["levels"] = summary["levels"]
    n = summary["candidate_count"]
    if arrays["membership"].shape != (n, levels) or arrays["local_market_available"].shape != (n,):
        raise ValueError("Invalid hierarchy dimensions")
    for i, (pool, coarse) in enumerate(zip(arrays["pools"], arrays["coarse"])):
        count = summary["levels"][i]["nodes"]
        ids = arrays["membership"][:, i]
        if (pool.shape != (count, n) or coarse.shape != (count, count)
                or (ids < 0).any() or (ids >= count).any()
                or not np.isfinite(pool.data).all() or not np.isfinite(coarse.data).all()):
            raise ValueError("Invalid hierarchy level")
    return arrays
