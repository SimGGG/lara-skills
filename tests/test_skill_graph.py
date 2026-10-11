"""Small independent examples for counting, threshold boundaries and cleanup."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.graphs import GraphThresholds, graph_metrics, select_graph
from lara_skills.postings import aggregate_cooccurrence, batch_incidence, clean_postings


class SkillGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = {"market_id": "TEST", "input_files": ["a.parquet", "b.parquet"],
                    "columns": {"source": "src", "posting_id": "id", "year": "yr", "month": "mo", "skills": "tags"},
                    "skill_separator": ",", "start_year": 2021, "end_year": 2022,
                    "skill_index": "index.parquet", "clean_output": "clean", "statistics_output": "stats",
                    "threads": 1, "memory_limit": "256MB", "batch_size": 2}
        dictionary = pd.DataFrame({"skill_code": list("ABCD"), "skill_name": list("ABCD"),
                                   "main": ["X"]*4, "sub": ["Y"]*4, "detail": ["Z"]*4,
                                   "embedding_row": range(4)})
        (self.root / "index.parquet").write_bytes(dictionary.to_parquet(index=False))
        columns = ["src", "id", "yr", "mo", "tags"]
        a = pd.DataFrame([
            ["s1", "p1", "2021", "1", "A,A,B,"], ["s1", "p1", "2021", "1", "B,A"],
            ["s1", "p2", "2021", "2", "B,C"], ["s1", "p3", "2025", "1", "A"],
            ["s1", "p4", "2021", "13", "A"], ["s1", None, "2021", "1", "A"],
            ["s1", "p5", "2021", "1", ""]], columns=columns)
        b = pd.DataFrame([
            ["s1", "p1", "2021", "1", "A,B"], ["s2", "p1", "2022", "1", "A,C"],
            ["s1", "p6", "2022", "1", "A,C"], ["s1", "p7", "2022", "1", "D"],
            ["s1", "p8", "2022", "1", None]], columns=columns)
        for name, frame in [("a", a), ("b", b)]:
            (self.root / f"{name}.parquet").write_bytes(frame.to_parquet(index=False))

    def test_clean_count_and_thresholds_end_to_end(self):
        clean = clean_postings(self.root, self.cfg)
        quality = json.loads((clean / "quality_summary.json").read_text())
        self.assertEqual(quality["postings"], 7)
        self.assertEqual(quality["duplicate_rows_removed"], 2)
        self.assertEqual(quality["posting_skill_links"], 9)
        self.assertEqual(quality["pair_events"], 4)
        stats = aggregate_cooccurrence(self.root, self.cfg)
        nodes = pd.read_parquet(io.BytesIO((stats / "nodes.parquet").read_bytes()))
        pairs = pd.read_parquet(io.BytesIO((stats / "pairs.parquet").read_bytes()))
        self.assertEqual(nodes.posting_count.tolist(), [3, 2, 3, 1])
        self.assertEqual(list(zip(pairs.source_node, pairs.target_node, pairs.cooccurrence)), [(0, 1, 1), (0, 2, 2), (1, 2, 1)])
        selected, edges = select_graph(nodes, pairs, GraphThresholds(min_cooccurrence=2, min_weight=2/3))
        self.assertEqual(len(edges), 1)  # Inclusive support and similarity cutoffs.
        metrics = graph_metrics(nodes, pairs, selected, edges)
        self.assertEqual(metrics["isolates"], 2)
        self.assertEqual(metrics["components"], 3)
        self.assertAlmostEqual(metrics["connected_frequency_fraction"], 6/9)
        selected, edges = select_graph(nodes, pairs, GraphThresholds(min_skill_postings=2, min_cooccurrence=1, min_weight=.9))
        self.assertEqual(graph_metrics(nodes, pairs, selected, edges)["isolates"], 3)
        selected, edges = select_graph(nodes, pairs, GraphThresholds(min_skill_postings=100))
        self.assertEqual(graph_metrics(nodes, pairs, selected, edges)["nodes"], 0)
        self.assertEqual(list(self.root.glob("*.partial_*")), [])

    def test_conflicting_duplicates_fail_without_partial_output(self):
        p = self.root / "b.parquet"
        frame = pd.read_parquet(io.BytesIO(p.read_bytes()))
        frame.loc[0, "tags"] = "C"
        p.write_bytes(frame.to_parquet(index=False))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            clean_postings(self.root, self.cfg)
        self.assertFalse((self.root / "clean").exists())
        self.assertEqual(list(self.root.glob("*.partial_*")), [])

    def test_incidence_is_binary_int64_and_respects_slices(self):
        skills = pa.array([["C"], *([["A", "A", "B"]] * 300), ["C"]]).slice(1, 300)
        matrix = batch_incidence(skills, pa.array(list("ABC")))
        counts = (matrix.T @ matrix).toarray()
        self.assertEqual(counts.dtype, np.int64)
        self.assertEqual(counts.tolist(), [[300, 300, 0], [300, 300, 0], [0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            batch_incidence(pa.array([["unknown"]]), pa.array(list("ABC")))

    def test_invalid_thresholds(self):
        for kwargs in [{"min_cooccurrence": 0}, {"min_skill_postings": 1.5},
                       {"min_weight": float("nan")}, {"min_weight": 1.1}, {"weight_metric": "unknown"}]:
            with self.assertRaises(ValueError):
                GraphThresholds(**kwargs)


if __name__ == "__main__":
    unittest.main()
