"""Toy data determinism, schema and the deliberate imperfections it must contain."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills import toy


class ToyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = toy.generate(toy.DEFAULT_SEED)

    def test_generation_is_deterministic_and_seed_sensitive(self):
        again = toy.generate(toy.DEFAULT_SEED)
        for key, frame in self.tables.items():
            self.assertEqual(toy.content_sha256(frame), toy.content_sha256(again[key]), key)
        other = toy.generate(toy.DEFAULT_SEED + 1)
        self.assertNotEqual(toy.content_sha256(self.tables["inputs/patents/labels.parquet"]),
                            toy.content_sha256(other["inputs/patents/labels.parquet"]))

    def test_dictionary_occupations_and_patent_tables(self):
        skills = self.tables["inputs/skills.csv"]
        self.assertEqual(len(skills), 240)
        self.assertTrue(skills.skill_code.is_unique and skills.skill_name.is_unique)
        self.assertEqual(list(skills.columns), ["skill_code", "skill_name", "main_group", "sub_group", "detail_group"])
        occupations = self.tables["inputs/occupations.csv"]
        self.assertEqual(len(occupations), 20)
        self.assertTrue(occupations.occupation_code.str.fullmatch(r"\d{4}").all())
        patents = self.tables["inputs/patents/patents.parquet"]
        labels = self.tables["inputs/patents/labels.parquet"]
        pool = self.tables["inputs/patents/candidate_pool.parquet"]
        tasks = self.tables["inputs/patents/task_status.parquet"]
        self.assertEqual(len(patents), 60)
        self.assertEqual(patents.application_year.value_counts().to_dict(), {y: 12 for y in range(2019, 2024)})
        self.assertEqual(patents.family_id.value_counts().get("FAM-0001"), 2)
        self.assertTrue(labels.relevance_score.isin([0, 1, 2]).all())
        self.assertTrue(pool.relevance_score.isna().any())
        self.assertEqual(len(pool) - len(labels), int(pool.relevance_score.isna().sum()))
        self.assertTrue(labels.skill_code.isin(skills.skill_code).all())
        self.assertEqual(len(tasks), 60)
        self.assertTrue((labels.groupby("sample_id").relevance_score.max() == 2).all())
        for text in patents.technology_description.head(5):
            self.assertIn("The invention relates to", text)

    def test_postings_contain_cleaning_cases_and_rare_skills(self):
        files = {k: v for k, v in self.tables.items() if k.startswith("inputs/postings/")}
        self.assertEqual(sorted(files), [f"inputs/postings/postings_{y}.parquet" for y in [2021, 2022, 2023, 2024]])
        frame = pd.concat(files.values(), ignore_index=True)
        self.assertEqual(list(frame.columns), ["source", "posting_id", "year", "month", "occupation", "skills"])
        self.assertGreater(len(frame), 24000)
        self.assertTrue(frame.posting_id.isna().any())
        self.assertTrue((frame.month == 13).any())
        self.assertTrue((frame.year == 2025).any())
        valid = frame[frame.posting_id.notna()]
        self.assertTrue(valid.duplicated(["source", "posting_id"]).any())
        counts = Counter(code for cell in frame.skills for code in cell.split(","))
        skills = self.tables["inputs/skills.csv"]
        unobserved, low = toy.rare_sets(24)
        for i in unobserved:
            self.assertNotIn(skills.skill_code.iloc[i], counts)
        for i in low:
            self.assertTrue(1 <= counts[skills.skill_code.iloc[i]] <= 4)
        self.assertEqual(frame.occupation.str.len().max(), 4)

    def test_writer_records_content_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = toy.write_toy_inputs(tmp, toy.DEFAULT_SEED)
            root = Path(tmp)
            self.assertTrue((root / "lara_app.json").is_file())
            config = json.loads((root / "lara_app.json").read_text(encoding="utf-8"))
            self.assertEqual(config["encoder"]["backend"], "hashed_ngram")
            for key, info in manifest["tables"].items():
                self.assertEqual(info["content_sha256"], toy.content_sha256(self.tables[key]))
                self.assertTrue((root / key).is_file())
            with self.assertRaises(FileExistsError):
                toy.write_toy_inputs(tmp, toy.DEFAULT_SEED)


if __name__ == "__main__":
    unittest.main()
