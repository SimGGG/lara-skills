"""Frozen text encoders shared by skill, occupation and patent encoding.

Two backends produce unit-length float32 vectors with the same interface:

* ``sentence_transformers`` (default): a pinned Sentence-Transformers model such as
  BGE-M3, as used in the paper. Requires ``pip install lara-skills[bge]``.
* ``hashed_ngram``: a deterministic hashed n-gram encoder with no external model.
  It powers the toy quickstart and continuous integration; it carries no learned
  semantics and must not be used to judge recommendation quality.
"""

import hashlib
import math
import re
from collections import Counter

import numpy as np

HASHED_NGRAM_MODEL_ID = "lara-skills/hashed-ngram"
HASHED_NGRAM_REVISION = hashlib.sha1(b"lara-skills hashed-ngram features v1").hexdigest()
BACKENDS = {"sentence_transformers", "hashed_ngram"}


def validate_vectors(vectors, rows, dimension):
    if vectors.shape != (rows, dimension) or vectors.dtype != np.float32:
        raise ValueError(f"Invalid feature shape/dtype: {vectors.shape}, {vectors.dtype}")
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embedding values")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise ValueError("Embeddings are not unit normalized")
    return {"shape": list(vectors.shape), "dtype": str(vectors.dtype),
            "finite": True, "norm_min": float(norms.min()), "norm_max": float(norms.max())}


def _check_texts(texts):
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Descriptions must be nonempty strings")


class SentenceTransformerEncoder:
    """Pinned, frozen dense encoder (CLS pooling, L2 normalization, no truncation)."""

    backend = "sentence_transformers"

    def __init__(self, config, device=None):
        import torch
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("The sentence_transformers backend needs `pip install lara-skills[bge]`") from error
        self.config = dict(config)
        self.device = device or config["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select device cpu or install a CUDA build of torch")
        torch.set_num_threads(config.get("cpu_threads", 8))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = SentenceTransformer(config["model_id"], revision=config["revision"],
                                         device=self.device, trust_remote_code=False,
                                         model_kwargs={"dtype": torch.float32})
        self.model.max_seq_length = config["max_seq_length"]
        self.model.eval().requires_grad_(False)
        self.pooling = self.model[1].get_config_dict()
        if self.pooling.get("pooling_mode") != "cls":
            raise ValueError(f"Unexpected pooling: {self.pooling}")
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        expected = config.get("dimension")
        if expected is not None and int(expected) != self.dimension:
            raise ValueError(f"Configured dimension {expected} differs from the model dimension {self.dimension}")

    def encode(self, texts, progress=False):
        _check_texts(texts)
        lengths = []
        for offset in range(0, len(texts), 64):
            tokenized = self.model.tokenizer(texts[offset:offset + 64], padding=False, truncation=False)
            lengths.extend(len(x) for x in tokenized["input_ids"])
        if max(lengths) > self.config["max_seq_length"]:
            raise ValueError(f"Input exceeds {self.config['max_seq_length']} tokens; no silent truncation")
        vectors = self.model.encode(texts, batch_size=self.config["batch_size"], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=progress)
        vectors = np.asarray(vectors, dtype=np.float32)
        validate_vectors(vectors, len(texts), self.dimension)
        return vectors, lengths

    def describe(self):
        return {"backend": self.backend, "model_id": self.config["model_id"], "revision": self.config["revision"],
                "pooling": self.pooling, "dimension": self.dimension}


class HashedNgramEncoder:
    """Deterministic bag of hashed word unigrams, word bigrams and character trigrams."""

    backend = "hashed_ngram"

    def __init__(self, config, device=None):
        self.config = dict(config)
        self.device = device or config.get("device", "cpu")
        if config.get("model_id") != HASHED_NGRAM_MODEL_ID or config.get("revision") != HASHED_NGRAM_REVISION:
            raise ValueError(f"hashed_ngram configs must set model_id={HASHED_NGRAM_MODEL_ID!r} "
                             f"and revision={HASHED_NGRAM_REVISION!r}")
        self.dimension = int(config.get("dimension", 1024))
        self.max_seq_length = int(config.get("max_seq_length", 8192))
        if self.dimension < 16 or self.max_seq_length < 1:
            raise ValueError("Invalid hashed encoder dimension or length limit")

    @staticmethod
    def features(text):
        words = re.findall(r"\w+", text.casefold())
        result = list(words)
        result.extend(a + " " + b for a, b in zip(words, words[1:]))
        for word in words:
            padded = f"#{word}#"
            result.extend(padded[i:i + 3] for i in range(len(padded) - 2))
        return words, result

    def vector(self, text):
        words, features = self.features(text)
        if not features:
            raise ValueError("Text has no alphanumeric tokens to encode")
        vector = np.zeros(self.dimension, dtype=np.float64)
        for feature, count in Counter(features).items():
            h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(h[:4], "little") % self.dimension
            sign = 1.0 if h[4] & 1 else -1.0
            vector[index] += sign * math.log1p(count)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Hashed features cancelled out; use a longer text")
        return (vector / norm).astype(np.float32), len(words)

    def encode(self, texts, progress=False):
        _check_texts(texts)
        vectors, lengths = [], []
        for text in texts:
            vector, length = self.vector(text)
            vectors.append(vector)
            lengths.append(length)
        if max(lengths) > self.max_seq_length:
            raise ValueError(f"Input exceeds {self.max_seq_length} words; no silent truncation")
        vectors = np.stack(vectors)
        validate_vectors(vectors, len(texts), self.dimension)
        return vectors, lengths

    def describe(self):
        return {"backend": self.backend, "model_id": HASHED_NGRAM_MODEL_ID, "revision": HASHED_NGRAM_REVISION,
                "dimension": self.dimension, "features": "word unigrams, word bigrams, character trigrams; signed hashing",
                "semantic": False}


def make_encoder(config, device=None):
    backend = config.get("backend", "sentence_transformers")
    if backend == "sentence_transformers":
        return SentenceTransformerEncoder(config, device)
    if backend == "hashed_ngram":
        return HashedNgramEncoder(config, device)
    raise ValueError(f"Unknown encoder backend {backend!r}; expected one of {sorted(BACKENDS)}")
