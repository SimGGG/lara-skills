"""Frozen query encoding of technology descriptions, aligned with the skill features."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .encoders import make_encoder
from .pipeline_io import digest, output_run, write_json, write_parquet
from .training_inputs import read_frame, verify_files


def query_text(description, perspective, config):
    if config.get("task_mode") == "single" and (config["perspectives"] != ["skills"] or config["query_template"] != "{description}"):
        raise ValueError("Single-task encoding requires plain descriptions and one internal task key")
    if perspective not in config["perspectives"]:
        raise ValueError("Unknown perspective")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Technology description is empty")
    return config["query_template"].format(description=description, perspective=perspective)


def generate_patent_embeddings(root, config):
    root = Path(root)
    skill_manifest = json.loads((root / config["skill_embedding_dir"] / "manifest.json").read_text(encoding="utf-8"))
    for key in ["model_id", "revision", "max_seq_length"]:
        if config[key] != skill_manifest["config"][key]:
            raise ValueError(f"Patent and skill encoders differ: {key}")
    source = root / config["input_path"]
    source_hash = digest(source)
    patents = read_frame(source).sort_values(config["id_column"]).reset_index(drop=True)
    if not patents[config["id_column"]].is_unique:
        raise ValueError("Duplicate patent IDs")
    patents = patents.rename(columns={config["id_column"]: "sample_id"})
    verify_files(root / config["split_dir"], ["assignments.parquet"])
    assignments = read_frame(root / config["split_dir"] / "assignments.parquet")
    patents = patents.merge(assignments[["sample_id", "split"]], on="sample_id", how="left", validate="one_to_one")
    if patents.split.isna().any():
        raise ValueError("Unassigned patent")
    with output_run(root, root / config["output_dir"], "patent_embeddings", config) as (out, manifest):
        torch.manual_seed(config["seed"])
        encoder = make_encoder(config)
        plain = patents[config["text_column"]].tolist()
        rows = []
        for _, patent in patents.iterrows():
            for perspective_id, perspective in enumerate(config["perspectives"]):
                rows.append({"sample_id": patent["sample_id"], "perspective": perspective,
                             "perspective_id": perspective_id, "split": patent.split,
                             "embedding_text": query_text(patent[config["text_column"]], perspective, config)})
        queries = pd.DataFrame(rows)
        queries.insert(0, "query_row", np.arange(len(queries)))
        print(f"Encoding {len(plain)} plain descriptions and {len(queries)} perspective queries", flush=True)
        all_texts = plain + queries.embedding_text.tolist()
        vectors, lengths = encoder.encode(all_texts, progress=True)
        probes = [0, len(plain), len(all_texts) - 1]
        checked, _ = encoder.encode([all_texts[i] for i in probes])
        error = float(np.abs(checked - vectors[probes]).max())
        if error > 1e-5:
            raise ValueError(f"Encoding/row alignment check failed: {error}")
        patent_index = patents[["sample_id", "split", config["text_column"]]].copy()
        patent_index.insert(0, "patent_row", np.arange(len(patents)))
        patent_index["token_count"] = lengths[:len(patents)]
        queries["token_count"] = lengths[len(patents):]
        for name, matrix in [("patent_embeddings.npy", vectors[:len(patents)]), ("query_embeddings.npy", vectors[len(patents):])]:
            buffer = io.BytesIO()
            np.save(buffer, matrix, allow_pickle=False)
            (out / name).write_bytes(buffer.getvalue())
            if not np.array_equal(matrix, np.load(io.BytesIO((out / name).read_bytes()), allow_pickle=False)):
                raise ValueError("Saved embedding differs from computed matrix")
        write_parquet(patent_index, out / "patent_index.parquet")
        write_parquet(queries, out / "query_index.parquet")
        write_json(out / "encoder_config.json", config)
        if digest(source) != source_hash:
            raise ValueError("Patent source changed during encoding")
        summary = {"patents": len(patents), "perspective_queries": len(queries), "dimension": int(vectors.shape[1]),
                   "dtype": "float32", "normalization": "L2", "min_tokens": min(lengths), "max_tokens": max(lengths),
                   "truncated_inputs": 0, "reencoding_max_abs_error": error, "labels_used": False,
                   "encoder_finetuned": False, "test_text_handling": "frozen encoding only; excluded from training and model selection"}
        write_json(out / "validation.json", summary)
        manifest.update({"summary": summary, "source_sha256": source_hash,
                         "skill_manifest_sha256": digest(root / config["skill_embedding_dir"] / "manifest.json"),
                         "split_manifest_sha256": digest(root / config["split_dir"] / "manifest.json"),
                         "torch_version": torch.__version__, "cuda": torch.version.cuda,
                         "device": torch.cuda.get_device_name(0) if encoder.device == "cuda" else "cpu"})
        print(json.dumps(summary), flush=True)
