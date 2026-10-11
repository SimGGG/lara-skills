"""Toy pipeline from raw inputs to recommendations, including resume and the variant toggle."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills import toy
from lara_skills.app import build, load_app_config, stage_config, workspace_paths
from lara_skills.evaluate import evaluate_split
from lara_skills.recommender import RecommendationEngine


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        toy.write_toy_inputs(cls.root, toy.DEFAULT_SEED)
        config = json.loads((cls.root / "lara_app.json").read_text(encoding="utf-8"))
        config["model"]["epochs"] = 3
        (cls.root / "lara_app.json").write_text(json.dumps(config), encoding="utf-8")
        cls.app = load_app_config(cls.root / "lara_app.json")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.outputs = build(cls.root, cls.app, "lara", "cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_stage_completed_with_package_provenance(self):
        for stage, relative in self.outputs.items():
            manifest = json.loads((self.root / relative / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed", stage)
            self.assertNotIn("entrypoint_sha256", manifest)
            self.assertEqual(manifest["package"]["name"], "lara-skills")
            self.assertIn("code_hashes", manifest)
        quality = json.loads((self.root / self.outputs["clean-postings"] / "quality_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(quality["postings"], 24000)
        self.assertGreater(quality["duplicate_rows_removed"], 0)
        candidates = pd.read_parquet(self.root / self.outputs["candidates"] / "candidates.parquet")
        self.assertEqual(len(candidates), 240)
        self.assertEqual(set(candidates.graph_status), {"connected", "isolated", "unobserved"})
        splits = json.loads((self.root / self.outputs["splits"] / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(splits["patents"], 60)
        self.assertEqual(splits["groups"], 58)

    def test_training_starts_at_the_semantic_baseline_and_engine_scores_everything(self):
        model_dir = self.root / self.outputs["train"]
        history = pd.read_csv(model_dir / "hgt_conditional_history.csv")
        semantic = pd.read_csv(model_dir / "semantic_retrieval.csv")
        row = semantic[(semantic.positive_grade_min == 1) & (semantic.k == 5)].iloc[0]
        self.assertAlmostEqual(history.iloc[0].validation_known_recall_at_5, row.recall_macro)
        engine = RecommendationEngine(self.root, self.outputs["train"], device="cpu")
        rows = engine.recommend_table(toy.SAMPLE_DESCRIPTION, 5)
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3, 4, 5])
        self.assertTrue(any("laser welding" in r["skill_name"] for r in rows))
        names = engine.recommend(toy.SAMPLE_DESCRIPTION, 3)
        self.assertEqual(names, [r["skill_name"] for r in rows[:3]])
        with self.assertRaises(ValueError):
            engine.recommend_table(toy.SAMPLE_DESCRIPTION, 0)

    def test_evaluation_covers_validation_and_explicit_test_and_rebuild_skips(self):
        config = stage_config(self.app, "evaluate", "lara", "cpu")
        with contextlib.redirect_stdout(io.StringIO()):
            validation = evaluate_split(self.root, {**config, "split": "validation",
                                                    "output_dir": workspace_paths(self.app, "lara")["evaluation"] + "_validation"})
            test = evaluate_split(self.root, {**config, "split": "test",
                                              "output_dir": workspace_paths(self.app, "lara")["evaluation"] + "_test"})
        self.assertEqual(validation["queries"], 15)
        self.assertEqual(test["queries"], 10)
        self.assertTrue(test["test_labels_read"])
        self.assertTrue(0 <= validation["known_positive_recall_at_5"] <= 1)
        scores = np.load(self.root / (workspace_paths(self.app, "lara")["evaluation"] + "_test") / "all_scores.npy")
        self.assertEqual(scores.shape, (10, 240))
        runs_before = len(list((self.root / "artifacts/runs").iterdir()))
        with contextlib.redirect_stdout(io.StringIO()):
            again = build(self.root, self.app, "lara", "cpu")
        self.assertEqual(again, self.outputs)
        self.assertEqual(len(list((self.root / "artifacts/runs").iterdir())), runs_before)

    def test_variant_toggle_trains_lara_na_on_the_same_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            outputs = build(self.root, self.app, "lara-na", "cpu")
        self.assertEqual(outputs["train"], "artifacts/models/toy_v1_lara-na")
        self.assertTrue((self.root / outputs["train"] / "hgt_network.pt").is_file())
        for stage in outputs:
            if stage != "train":
                self.assertEqual(outputs[stage], self.outputs[stage])
        engine = RecommendationEngine(self.root, outputs["train"], device="cpu")
        self.assertEqual(engine.model_name, "hgt_network")
        self.assertEqual(len(engine.recommend_table(toy.SAMPLE_DESCRIPTION, 4)), 4)


if __name__ == "__main__":
    unittest.main()
