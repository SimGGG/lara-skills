"""Application config defaults, per-stage composition and the command-line parser."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills import cli, toy
from lara_skills.app import STAGES, load_app_config, output_dir, stage_config, workspace_paths


class AppConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "lara_app.json"
        self.config_path.write_text(json.dumps(toy.app_config(7)), encoding="utf-8")

    def test_defaults_are_filled_and_stages_compose(self):
        app = load_app_config(self.config_path)
        self.assertEqual(app["model"]["layers"], 2)
        self.assertEqual(app["model"]["hidden_dim"], 64)
        self.assertEqual(app["postings"]["threads"], 2)
        self.assertEqual(app["patents"]["columns"]["id"], "sample_id")
        for stage in STAGES:
            config = stage_config(app, stage)
            self.assertIsInstance(config, dict)
            self.assertTrue(output_dir(app, stage).startswith(("data/processed/toy_v1", "artifacts/")))
        train = stage_config(app, "train")
        self.assertEqual(train["training_models"], ["hgt_conditional"])
        self.assertEqual(train["task_mode"], "single")
        self.assertEqual(train["top_k"], 5)
        self.assertEqual(stage_config(app, "train", variant="lara-na")["training_models"], ["hgt_network"])
        self.assertEqual(workspace_paths(app, "lara-na")["model"], "artifacts/models/toy_v1_lara-na")
        patents = stage_config(app, "patent-embeddings")
        self.assertEqual(patents["perspectives"], ["skills"])
        self.assertEqual(patents["query_template"], "{description}")
        self.assertEqual(stage_config(app, "skill-embeddings", device="cuda")["device"], "cuda")

    def test_invalid_configs_are_rejected(self):
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        del raw["skills"]
        broken = self.root / "broken.json"
        broken.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "skills"):
            load_app_config(broken)
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["patents"]["columns"]["id"] = "patent_id"
        broken.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sample_id"):
            load_app_config(broken)
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw["model"]["variant"] = "unknown"
        broken.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "variant"):
            load_app_config(broken)
        with self.assertRaises(ValueError):
            stage_config(load_app_config(self.config_path), "mystery")

    def test_parser_entry_points(self):
        with self.assertRaises(SystemExit) as exit_info:
            cli.main(["--version"])
        self.assertEqual(exit_info.exception.code, 0)
        with self.assertRaises(SystemExit):
            cli.main(["build"])
        with self.assertRaises(SystemExit):
            cli.main(["recommend", "--config", str(self.config_path)])


if __name__ == "__main__":
    unittest.main()
