"""Deterministic hashed encoder contract and encoder factory validation."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lara_skills.encoders import (HASHED_NGRAM_MODEL_ID, HASHED_NGRAM_REVISION, HashedNgramEncoder,
                                  make_encoder, validate_vectors)

CONFIG = {"backend": "hashed_ngram", "model_id": HASHED_NGRAM_MODEL_ID, "revision": HASHED_NGRAM_REVISION,
          "dimension": 256, "max_seq_length": 8, "device": "cpu"}


class EncoderTests(unittest.TestCase):
    def test_hashed_vectors_are_unit_length_deterministic_and_text_sensitive(self):
        encoder = make_encoder(CONFIG)
        self.assertIsInstance(encoder, HashedNgramEncoder)
        first, lengths = encoder.encode(["laser welding fixture", "laser welding setup", "powder coating cost"])
        second, _ = encoder.encode(["laser welding fixture", "laser welding setup", "powder coating cost"])
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, 256))
        self.assertEqual(lengths, [3, 3, 3])
        validate_vectors(first, 3, 256)
        similar, unrelated = float(first[0] @ first[1]), float(first[0] @ first[2])
        self.assertGreater(similar, unrelated)
        self.assertFalse(np.allclose(first[0], first[1]))
        self.assertEqual(len(HASHED_NGRAM_REVISION), 40)
        self.assertEqual(encoder.describe()["semantic"], False)

    def test_features_include_bigrams_and_character_trigrams(self):
        words, features = HashedNgramEncoder.features("Laser Welding")
        self.assertEqual(words, ["laser", "welding"])
        self.assertIn("laser welding", features)
        self.assertIn("#la", features)
        self.assertIn("ng#", features)

    def test_invalid_inputs_and_configs_are_rejected(self):
        encoder = make_encoder(CONFIG)
        with self.assertRaises(ValueError):
            encoder.encode([])
        with self.assertRaises(ValueError):
            encoder.encode(["ok", "   "])
        with self.assertRaisesRegex(ValueError, "alphanumeric"):
            encoder.encode(["!!! ???"])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            encoder.encode(["one two three four five six seven eight nine"])
        with self.assertRaisesRegex(ValueError, "revision"):
            make_encoder({**CONFIG, "revision": "0" * 40})
        with self.assertRaisesRegex(ValueError, "Unknown encoder backend"):
            make_encoder({**CONFIG, "backend": "mystery"})
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            validate_vectors(np.ones((2, 4), dtype=np.float32), 2, 4)


if __name__ == "__main__":
    unittest.main()
