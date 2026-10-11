"""Exercise temporary-output cleanup of skill encoding without any real encoder."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills import skill_embeddings


class FakeEncoder:
    backend = "fake"
    dimension = 2
    device = "cpu"

    def encode(self, texts, progress=False):
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(texts), 1)), [2] * len(texts)

    def describe(self):
        return {"backend": self.backend, "dimension": self.dimension}


class EmbeddingCleanupTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        self.destination = self.root / "data/processed/features"
        self.cfg = {"dtype": "float32", "normalize_embeddings": True, "revision": "a" * 40, "backend": "fake",
                    "device": "cpu", "output_dir": "data/processed/features", "seed": 42, "cpu_threads": 1,
                    "input_path": "input.csv", "model_id": "mock-model", "max_seq_length": 8192, "batch_size": 2,
                    "dimension": 2, "id_column": "skill_code", "name_column": "skill_name",
                    "taxonomy_columns": ["main"], "text_template": "{skill_name} {taxonomy_path}", "taxonomy_separator": " > "}
        (self.root / "input.csv").write_bytes(b"unchanged fixture input")
        self.frame = pd.DataFrame({"embedding_row": [0, 1, 2], "skill_code": ["a", "b", "c"],
                                   "skill_name": ["A", "B", "C"], "embedding_text": ["A", "B", "C"]})
        for target, replacement in [("prepare_inputs", lambda *a: self.frame.copy()),
                                    ("make_encoder", lambda cfg, device=None: FakeEncoder())]:
            patcher = patch.object(skill_embeddings, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def check_failed_run(self, status):
        self.assertFalse(self.destination.exists())
        self.assertEqual(list((self.root / "data/processed").glob("*.partial_*")), [])
        manifests = list((self.root / "artifacts/runs").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(pd.read_json(manifests[0], typ="series")["status"], status)
        self.assertEqual((self.root / "input.csv").read_bytes(), b"unchanged fixture input")

    def test_encoder_failure_creates_no_output_directory(self):
        with patch.object(skill_embeddings, "make_encoder", side_effect=RuntimeError("load failed")):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                skill_embeddings.generate(self.root, self.cfg)
        self.check_failed_run("failed")
        self.assertFalse(self.destination.parent.exists())

    def test_write_failure_cleans_partial_files_but_preserves_other_runs(self):
        other = self.destination.parent / "other.partial_unrelated"
        other.mkdir(parents=True)
        (other / "keep.txt").write_text("keep", encoding="utf-8")
        with patch.object(pd.DataFrame, "to_parquet", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                skill_embeddings.generate(self.root, self.cfg)
        self.assertEqual(list(self.destination.parent.glob("features.partial_*")), [])
        self.assertEqual((other / "keep.txt").read_text(encoding="utf-8"), "keep")
        self.assertFalse(self.destination.exists())

    def test_keyboard_interrupt_cleans_partial_files(self):
        with patch.object(pd.DataFrame, "to_parquet", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                skill_embeddings.generate(self.root, self.cfg)
        self.check_failed_run("interrupted")

    def test_success_retains_completed_output_only(self):
        result = skill_embeddings.generate(self.root, self.cfg)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["encoder"]["backend"], "fake")
        self.assertEqual(result["package"]["name"], "lara-skills")
        self.assertIn("code_hashes", result)
        self.assertTrue((self.destination / "skill_embeddings.npy").is_file())
        self.assertEqual(list(self.destination.parent.glob("*.partial_*")), [])

    def test_existing_output_is_not_overwritten(self):
        self.destination.mkdir(parents=True)
        marker = self.destination / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            skill_embeddings.generate(self.root, self.cfg)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((self.root / "artifacts").exists())


if __name__ == "__main__":
    unittest.main()
