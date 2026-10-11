"""Frozen skill-dictionary features with explicit ID alignment and provenance."""

import hashlib
import io
import json
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ._version import __version__
from .encoders import make_encoder, validate_vectors
from .pipeline_io import PACKAGE_NAME, code_hashes, digest, environment, read_table, write_json


def prepare_inputs(path, cfg):
    columns = [cfg["id_column"], cfg["name_column"], *cfg["taxonomy_columns"]]
    table = read_table(path, cfg.get("sheet_name"))
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"Skill dictionary lacks columns: {missing}")
    frame = table[columns].copy()
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"Missing values in {column}")
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Blank values in {column}")
    if frame.empty or frame[cfg["id_column"]].duplicated().any():
        raise ValueError("Empty dictionary or duplicate skill IDs")
    frame = frame.sort_values(cfg["id_column"], kind="stable").reset_index(drop=True)
    frame = frame.rename(columns={cfg["id_column"]: "skill_code", cfg["name_column"]: "skill_name"})
    frame.insert(0, "embedding_row", np.arange(len(frame), dtype=np.int64))
    frame["taxonomy_path"] = frame[cfg["taxonomy_columns"]].agg(cfg["taxonomy_separator"].join, axis=1)
    frame["embedding_text"] = [cfg["text_template"].format(skill_name=name, taxonomy_path=path)
                               for name, path in zip(frame.skill_name, frame.taxonomy_path)]
    frame["text_sha256"] = frame.embedding_text.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    return frame


def generate(root, config):
    """Encode the dictionary once; refuse to overwrite and clean this run's partial output on failure."""
    root = Path(root)
    cfg = json.loads(Path(config).read_text(encoding="utf-8")) if isinstance(config, (str, Path)) else dict(config)
    if cfg.get("dtype", "float32") != "float32" or not cfg.get("normalize_embeddings", True):
        raise ValueError("This pipeline requires float32 and L2 normalization")
    if len(str(cfg["revision"])) != 40:
        raise ValueError("Pin a full 40-character model revision")
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select device cpu")
    destination = root / cfg["output_dir"]
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_skill_embeddings"
    run_dir = root / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    temporary_output = None
    manifest = {"run_id": run_id, "purpose": "skill_embeddings", "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config": cfg, "output_dir": cfg["output_dir"], "training": False,
                "labels_used": False, "graph_constructed": False,
                "package": {"name": PACKAGE_NAME, "version": __version__}, "command": list(sys.argv)}
    try:
        write_json(run_dir / "manifest.json", manifest)
        write_json(run_dir / "config.json", cfg)
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.set_num_threads(cfg.get("cpu_threads", 8))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        input_path = root / cfg["input_path"]
        input_hash = digest(input_path)
        frame = prepare_inputs(input_path, cfg)
        print(f"Validated {len(frame)} unique skills. Loading the frozen encoder.", flush=True)
        encoder = make_encoder(cfg)
        dimension = int(cfg.get("dimension", encoder.dimension))
        if dimension != encoder.dimension:
            raise ValueError(f"Configured dimension {dimension} differs from the encoder dimension {encoder.dimension}")
        texts = frame.embedding_text.tolist()
        vectors, counts = encoder.encode(texts, progress=True)
        frame["token_count"] = counts
        validation = validate_vectors(vectors, len(frame), dimension)
        # Independently re-encode separated rows to check batching and row alignment.
        probes = [0, len(frame) // 2, len(frame) - 1]
        repeated, _ = encoder.encode([texts[i] for i in probes])
        difference = float(np.abs(vectors[probes] - repeated).max())
        if difference > 1e-5:
            raise ValueError(f"Re-encoding/row alignment check failed: {difference}")
        validation.update({"reencoded_rows": probes, "reencode_max_absolute_difference": difference,
                           "token_count_min": int(min(counts)), "token_count_max": int(max(counts)),
                           "truncated_rows": 0, "duplicate_ids": 0,
                           "duplicate_input_texts": int(frame.embedding_text.duplicated().sum())})
        buffer = io.BytesIO()
        np.save(buffer, vectors, allow_pickle=False)
        # Allocate staging only when ready to write, and always clean this run's
        # temporary output on failure/interruption. Never sweep other runs.
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = tempfile.TemporaryDirectory(prefix=destination.name + ".partial_", dir=destination.parent)
        staging = Path(temporary_output.name)
        (staging / "skill_embeddings.npy").write_bytes(buffer.getvalue())
        (staging / "skill_index.parquet").write_bytes(frame.to_parquet(index=False))
        frame.to_csv(staging / "skill_index.csv", index=False, encoding="utf-8-sig")
        saved_vectors = np.load(io.BytesIO((staging / "skill_embeddings.npy").read_bytes()), allow_pickle=False)
        saved_index = pd.read_parquet(io.BytesIO((staging / "skill_index.parquet").read_bytes()))
        validate_vectors(saved_vectors, len(frame), dimension)
        pd.testing.assert_frame_equal(frame, saved_index)
        if not np.array_equal(vectors, saved_vectors):
            raise ValueError("Saved vectors differ from computed vectors")
        if input_hash != digest(input_path):
            raise ValueError("Input dictionary changed during generation")
        validation["saved_roundtrip_verified"] = True
        validation["input_hash_unchanged"] = True
        # Descriptive inspection only: these neighbors are not relevance labels or evaluation scores.
        examples = []
        for i in probes:
            scores = vectors @ vectors[i]
            scores[i] = -np.inf
            neighbors = np.argsort(-scores, kind="stable")[:5]
            examples.append({"skill_code": frame.iloc[i].skill_code, "skill_name": frame.iloc[i].skill_name,
                             "neighbors": [{"skill_code": frame.iloc[j].skill_code, "skill_name": frame.iloc[j].skill_name,
                                            "cosine_similarity": float(scores[j])} for j in neighbors]})
        write_json(run_dir / "neighbor_inspection.json", {"purpose": "descriptive_only_not_accuracy_evaluation", "examples": examples})
        write_json(staging / "validation.json", validation)
        write_json(staging / "config.json", cfg)
        manifest.update({"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(),
                         "input": {"path": cfg["input_path"], "sha256": input_hash},
                         "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
                         "encoder": encoder.describe(), "validation": validation,
                         "environment": {**environment(), "cuda": torch.version.cuda,
                                         "device": torch.cuda.get_device_name(0) if cfg["device"] == "cuda" else "cpu"},
                         "code_hashes": code_hashes(),
                         "files": {p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
                                   for p in staging.iterdir() if p.is_file()}})
        write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"output_dir": str(destination), "run_id": run_id, "validation": validation}, ensure_ascii=False), flush=True)
        return manifest
    except BaseException as exc:
        manifest.update({"status": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                         "error": repr(exc), "finished_at": datetime.now(timezone.utc).isoformat()})
        write_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        if temporary_output is not None:
            temporary_output.cleanup()
