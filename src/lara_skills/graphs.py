"""Application-facing configurable edge filtering and structural diagnostics."""

import io
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .pipeline_io import digest, output_run, write_json, write_parquet


@dataclass(frozen=True)
class GraphThresholds:
    min_skill_postings: int = 1
    min_cooccurrence: int = 5
    weight_metric: str = "cosine"
    min_weight: float = 0.01
    min_active_years: int = 1

    def __post_init__(self):
        for name in ("min_skill_postings", "min_cooccurrence", "min_active_years"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.weight_metric not in {"cosine", "jaccard"}:
            raise ValueError("weight_metric must be cosine or jaccard")
        if not np.isfinite(self.min_weight) or not 0 <= self.min_weight <= 1:
            raise ValueError("min_weight must be between 0 and 1")


def select_graph(nodes, pairs, thresholds):
    """Return selected nodes (including isolates) and undirected, unique edges.

    Node IDs remain stable across thresholds and map to the cached statistics.
    All cutoffs are inclusive. No implicit top-k pruning or isolate removal.
    """
    t = thresholds
    if not np.array_equal(nodes.node_id.to_numpy(), np.arange(len(nodes))):
        raise ValueError("Statistics node IDs must be contiguous and sorted")
    eligible = nodes.posting_count.to_numpy() >= t.min_skill_postings
    source, target = pairs.source_node.to_numpy(), pairs.target_node.to_numpy()
    mask = (eligible[source] & eligible[target]
            & (pairs.cooccurrence.to_numpy() >= t.min_cooccurrence)
            & (pairs[t.weight_metric].to_numpy() >= t.min_weight)
            & (pairs.active_years.to_numpy() >= t.min_active_years))
    return nodes.loc[eligible].copy(), pairs.loc[mask].copy()


def graph_metrics(all_nodes, all_pairs, nodes, edges):
    n_all, n, m = len(all_nodes), len(nodes), len(edges)
    src, dst = edges.source_node.to_numpy(), edges.target_node.to_numpy()
    degree = np.bincount(np.concatenate([src, dst]), minlength=n_all)
    selected = nodes.node_id.to_numpy()
    degree_selected = degree[selected]
    graph = sparse.coo_matrix((np.ones(m, dtype=np.int8), (src, dst)), shape=(n_all, n_all)).tocsr()
    graph = graph[selected][:, selected]
    components, labels = connected_components(graph, directed=False)
    largest = int(np.bincount(labels).max()) if n else 0
    freq = all_nodes.posting_count.to_numpy()
    connected = degree > 0
    counts = edges.cooccurrence.to_numpy()
    return {"nodes": n, "edges": m, "removed_nodes": n_all - n,
            "isolates": int((degree_selected == 0).sum()),
            "connected_nodes": int(connected.sum()), "components": int(components),
            "largest_component_nodes": largest, "largest_component_fraction": largest / n if n else 0,
            "connected_skill_fraction": float(connected.mean()),
            "connected_frequency_fraction": float(freq[connected].sum() / freq.sum()),
            "edge_retention": m / len(all_pairs) if len(all_pairs) else 0,
            "pair_event_retention": float(counts.sum() / all_pairs.cooccurrence.sum()) if len(all_pairs) else 0,
            "density": 2 * m / (n * (n - 1)) if n > 1 else 0,
            "mean_degree": 2 * m / n if n else 0,
            "median_degree": float(np.median(degree_selected)) if n else 0,
            "p95_degree": float(np.quantile(degree_selected, .95)) if n else 0,
            "max_degree": int(degree_selected.max()) if n else 0,
            "edges_one_year_fraction": float((edges.active_years == 1).mean()) if m else 0,
            "rare_under10_connected": int((connected & (freq < 10)).sum()),
            "rare_under100_connected": int((connected & (freq < 100)).sum())}


def load_statistics(root, cfg):
    directory = Path(root) / cfg["statistics_dir"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Statistics are incomplete")
    for name in ["nodes.parquet", "pairs.parquet"]:
        if digest(directory / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Statistics integrity check failed: {name}")
    nodes = pd.read_parquet(io.BytesIO((directory / "nodes.parquet").read_bytes()))
    pairs = pd.read_parquet(io.BytesIO((directory / "pairs.parquet").read_bytes()))
    return directory, nodes, pairs


def evaluate_thresholds(root, cfg):
    directory, nodes, pairs = load_statistics(root, cfg)
    with output_run(root, Path(root) / cfg["evaluation_output"], "threshold_evaluation", cfg) as (out, manifest):
        results = []
        grid = cfg["grid"]
        for i, values in enumerate(itertools.product(grid["min_skill_postings"], grid["min_cooccurrence"], grid["min_weight"]), 1):
            params = {**cfg["thresholds"], **dict(zip(["min_skill_postings", "min_cooccurrence", "min_weight"], values))}
            t = GraphThresholds(**params)
            selected_nodes, edges = select_graph(nodes, pairs, t)
            metrics = graph_metrics(nodes, pairs, selected_nodes, edges)
            results.append({**asdict(t), **metrics})
            if i % 5 == 0:
                print(f"Threshold combinations evaluated: {i}", flush=True)
        table = pd.DataFrame(results)
        table.to_csv(out / "threshold_metrics.csv", index=False, encoding="utf-8-sig")
        write_parquet(table, out / "threshold_metrics.parquet")
        manifest.update({"statistics_manifest_sha256": digest(directory / "manifest.json"),
                         "combinations": len(table), "gold_labels_used": False,
                         "evaluation_type": "structural_diagnostics_not_prediction_accuracy"})
    return table


def build_graph(root, cfg):
    directory, all_nodes, all_pairs = load_statistics(root, cfg)
    thresholds = GraphThresholds(**cfg["thresholds"])
    nodes, edges = select_graph(all_nodes, all_pairs, thresholds)
    metrics = graph_metrics(all_nodes, all_pairs, nodes, edges)
    degree = np.bincount(np.concatenate([edges.source_node.to_numpy(), edges.target_node.to_numpy()]), minlength=len(all_nodes))
    nodes["degree"] = degree[nodes.node_id.to_numpy()]
    nodes["is_isolated"] = nodes.degree == 0
    edges["weight"] = edges[thresholds.weight_metric]
    with output_run(root, Path(root) / cfg["graph_output"], "build_skill_graph", cfg) as (out, manifest):
        write_parquet(nodes, out / "nodes.parquet")
        write_parquet(edges, out / "edges.parquet")
        write_json(out / "metrics.json", metrics)
        write_json(out / "thresholds.json", asdict(thresholds))
        manifest.update({"statistics_manifest_sha256": digest(directory / "manifest.json"),
                         "metrics": metrics, "edge_representation": "undirected_once_source_less_than_target",
                         "node_id_policy": "stable_statistics_id_not_necessarily_contiguous_after_filtering",
                         "isolated_node_policy": "retained", "graph_training_executed": False})
    print(json.dumps(metrics), flush=True)
    return metrics
