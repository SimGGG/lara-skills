"""Standardized occupation nodes and posting-derived bipartite demand relations."""

import io
import json
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from .hierarchy import row_normalize
from .pipeline_io import digest, output_run, read_table, write_json, write_parquet
from .training_inputs import read_frame, verify_files


def occupation_dictionary(frame, config):
    columns = config["path_columns"]
    paths = frame[columns].ffill()
    valid = frame[config["code_column"]].notna()
    result = pd.DataFrame({
        "occupation_code": frame.loc[valid, config["code_column"]].astype(str).str.strip().str.zfill(config["code_width"]),
        "occupation_name": frame.loc[valid, config["name_column"]],
    })
    if paths.loc[valid].isna().any().any() or result.isna().any().any():
        raise ValueError("Missing occupation name or taxonomy path")
    paths = paths.loc[valid].apply(lambda x: x.astype(str).str.strip())
    result["occupation_name"] = result.occupation_name.astype(str).str.strip()
    if (not result.occupation_code.is_unique or result.occupation_name.eq("").any()
            or paths.eq("").any().any()
            or not result.occupation_code.str.fullmatch(r"\d{" + str(config["code_width"]) + "}").all()):
        raise ValueError("Invalid or duplicate occupation dictionary keys")
    result["taxonomy_path"] = paths.agg(" > ".join, axis=1)
    result = result.sort_values("occupation_code").reset_index(drop=True)
    result.insert(0, "occupation_id", np.arange(len(result), dtype=np.int32))
    result["embedding_text"] = [config["embedding_template"].format(name=r.occupation_name, path=r.taxonomy_path)
                                for r in result.itertuples()]
    return result


def aggregate_postings(connection, paths):
    """Input contract: cleaned unique posting rows, deduplicated known skill lists."""
    connection.read_parquet([str(p) for p in paths]).project("occupation_code, skills").create_view("postings")
    counts = connection.execute("SELECT occupation_code, count(*) AS posting_count FROM postings GROUP BY 1").df()
    # Count postings without extracted skills in the occupation denominator too.
    skills = connection.execute("SELECT skill_code, count(*) AS posting_count FROM postings, "
                                "UNNEST(skills) AS t(skill_code) GROUP BY 1").df()
    pairs = connection.execute("SELECT occupation_code, skill_code, count(*) AS posting_count FROM postings, "
                               "UNNEST(skills) AS t(skill_code) GROUP BY 1,2 ORDER BY 1,2").df()
    return counts, skills, pairs


def occupation_edge_weights(edges, total_postings, weighting="cosine", prior_strength=100., support_shrinkage=10.):
    """Nonnegative demand weights; smoothing settings are independent of thresholds."""
    joint = edges.posting_count.to_numpy(dtype=np.float64)
    occupation = edges.occupation_posting_count.to_numpy(dtype=np.float64)
    skill = edges.skill_posting_count.to_numpy(dtype=np.float64)
    if (not np.isfinite(total_postings) or total_postings <= 0
            or not all(np.isfinite(x).all() for x in [joint, occupation, skill])
            or (joint <= 0).any() or (joint > occupation).any() or (joint > skill).any()
            or (occupation > total_postings).any() or (skill > total_postings).any()):
        raise ValueError("Invalid demand counts")
    if weighting == "cosine":
        return joint / np.sqrt(occupation * skill)
    if weighting == "frequency":
        return joint
    if weighting != "specialization":
        raise ValueError("Unknown occupation weighting")
    if not np.isfinite([prior_strength, support_shrinkage]).all() or min(prior_strength, support_shrinkage) <= 0:
        raise ValueError("Specialization smoothing must be finite and positive")
    prior = skill / total_postings
    conditional = (joint + prior_strength * prior) / (occupation + prior_strength)
    return joint / (joint + support_shrinkage) * np.maximum(0., np.log(conditional / prior))


def demand_relations(nodes, candidates, counts, skill_counts, pairs, minimum,
                     weighting="cosine", prior_strength=100., support_shrinkage=10.):
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("Minimum occupation-skill postings must be a positive integer")
    for frame, keys in [(counts, ["occupation_code"]), (skill_counts, ["skill_code"]),
                        (pairs, ["occupation_code", "skill_code"])]:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError("Missing or duplicate relation keys")
        if (frame.posting_count <= 0).any() or not np.equal(frame.posting_count, np.floor(frame.posting_count)).all():
            raise ValueError("Counts must be positive integers")
    if (not np.array_equal(nodes.occupation_id, np.arange(len(nodes)))
            or not nodes.occupation_code.is_unique or not candidates.skill_code.is_unique
            or not np.array_equal(candidates.candidate_id, np.arange(len(candidates)))):
        raise ValueError("Node IDs must match feature rows")
    if not set(counts.occupation_code) <= set(nodes.occupation_code):
        raise ValueError("Unknown or missing occupation code in postings")
    if not set(skill_counts.skill_code) <= set(candidates.skill_code):
        raise ValueError("Unknown skill code in postings")
    nodes = nodes.copy()
    nodes["posting_count"] = nodes.occupation_code.map(counts.set_index("occupation_code").posting_count).fillna(0).astype("int64")
    edges = pairs.copy()
    edges["occupation_id"] = edges.occupation_code.map(nodes.set_index("occupation_code").occupation_id)
    edges["candidate_id"] = edges.skill_code.map(candidates.set_index("skill_code").candidate_id)
    edges["occupation_posting_count"] = edges.occupation_code.map(counts.set_index("occupation_code").posting_count)
    edges["skill_posting_count"] = edges.skill_code.map(skill_counts.set_index("skill_code").posting_count)
    if (edges.isna().any().any() or (edges.posting_count > edges.occupation_posting_count).any()
            or (edges.posting_count > edges.skill_posting_count).any()):
        raise ValueError("Invalid relation marginals or unknown endpoints")
    edges["weight"] = occupation_edge_weights(edges, int(counts.posting_count.sum()), weighting, prior_strength, support_shrinkage)
    edges = edges[(edges.posting_count >= minimum) & (edges.weight > 0)].reset_index(drop=True)
    matrix = sparse.coo_matrix((edges.weight.to_numpy(dtype=np.float32),
                               (edges.occupation_id, edges.candidate_id)), shape=(len(nodes), len(candidates))).tocsr()
    nodes["degree"] = np.diff(matrix.indptr)
    return nodes, edges, row_normalize(matrix), row_normalize(matrix.T)


def prepare_occupations(root, config):
    from .encoders import make_encoder
    root = Path(root)
    directory = root / config["postings_dir"]
    names = [f"postings_{year}.parquet" for year in config["years"]]
    verify_files(directory, names)
    candidate_dir = root / config["candidate_dir"]
    verify_files(candidate_dir, ["candidates.parquet", "feature_reference.json"])
    candidates = read_frame(candidate_dir / "candidates.parquet")
    reference = json.loads((candidate_dir / "feature_reference.json").read_text(encoding="utf-8"))
    skill_manifest = json.loads((root / reference["path_relative_to_project_root"]).with_name("manifest.json").read_text(encoding="utf-8"))
    for key in ["model_id", "revision", "max_seq_length"]:
        if skill_manifest["config"][key] != config["encoder"][key]:
            raise ValueError("Occupation and skill encoders differ")
    source = root / config["dictionary_path"]
    nodes = occupation_dictionary(read_table(source, config.get("sheet_name")), config)
    with output_run(root, root / config["output_dir"], "occupation_skill_graph", config) as (out, manifest):
        with tempfile.TemporaryDirectory(prefix="occupation_spill_") as spill:
            con = duckdb.connect(config={"threads": config["threads"], "memory_limit": config["memory_limit"], "temp_directory": spill})
            try:
                counts, skill_counts, pairs = aggregate_postings(con, [directory / name for name in names])
            finally:
                con.close()
        observed = candidates.set_index("skill_code").posting_count.reindex(skill_counts.skill_code).to_numpy()
        if not np.array_equal(observed, skill_counts.posting_count):
            raise ValueError("Candidate and posting skill marginals differ")
        threshold_rows = []
        for threshold in sorted(set(config["threshold_grid"] + [config["min_occupation_skill_postings"]])):
            n, e, _, reverse = demand_relations(nodes, candidates, counts, skill_counts, pairs, threshold)
            connected = np.diff(reverse.indptr) > 0
            threshold_rows.append({"minimum": threshold, "edges": len(e), "connected_occupations": int((n.degree > 0).sum()),
                                   "connected_skills": int(connected.sum()), "isolated_skills": int((~connected).sum()),
                                   "skill_graph_isolates_with_occupation": int((connected & candidates.is_isolated.to_numpy()).sum())})
        nodes, edges, to_occ, to_skill = demand_relations(nodes, candidates, counts, skill_counts, pairs, config["min_occupation_skill_postings"])
        print(json.dumps(threshold_rows), flush=True)
        encoder = make_encoder(config["encoder"])
        vectors, lengths = encoder.encode(nodes.embedding_text.tolist(), progress=True)
        probes = [0, len(nodes) // 2, len(nodes) - 1]
        check, _ = encoder.encode(nodes.iloc[probes].embedding_text.tolist())
        error = float(np.abs(check - vectors[probes]).max())
        if error > 1e-5:
            raise ValueError("Occupation embedding alignment check failed")
        nodes["token_count"] = lengths
        for filename, frame in [("occupation_nodes.parquet", nodes), ("occupation_skill_counts.parquet", pairs),
                                ("occupation_skill_edges.parquet", edges), ("skill_counts.parquet", skill_counts)]:
            write_parquet(frame, out / filename)
        pd.DataFrame(threshold_rows).to_csv(out / "threshold_evaluation.csv", index=False)
        for name, value in [("skill_to_occupation", to_occ), ("occupation_to_skill", to_skill)]:
            buffer = io.BytesIO(); sparse.save_npz(buffer, value)
            (out / f"{name}.npz").write_bytes(buffer.getvalue())
        buffer = io.BytesIO(); np.save(buffer, vectors, allow_pickle=False)
        (out / "occupation_embeddings.npy").write_bytes(buffer.getvalue())
        write_json(out / "encoder_config.json", config["encoder"])
        write_json(out / "config.json", config)
        summary = {"market_id": config["market_id"], "years": config["years"], "postings": int(counts.posting_count.sum()),
                   "occupation_nodes": len(nodes), "observed_occupations": int((nodes.posting_count > 0).sum()),
                   "candidate_count": len(candidates), "raw_relations": len(pairs), "retained_relations": len(edges),
                   "min_occupation_skill_postings": config["min_occupation_skill_postings"],
                   "skill_isolates": int((np.diff(to_skill.indptr) == 0).sum()),
                   "occupation_isolates": int((nodes.degree == 0).sum()), "gold_labels_used": False,
                   "weight": "n_os / sqrt(n_o * n_s); separate receiver-row normalization per direction",
                   "embedding_shape": list(vectors.shape), "max_tokens": max(lengths), "reencoding_max_error": error}
        write_json(out / "summary.json", summary)
        manifest.update({"summary": summary, "candidate_manifest_sha256": digest(candidate_dir / "manifest.json"),
                         "postings_manifest_sha256": digest(directory / "manifest.json"), "dictionary_sha256": digest(source)})
        print(json.dumps(summary), flush=True)


def load_occupations(root, occupation_dir, candidate_dir):
    root, directory = Path(root), Path(root) / occupation_dir
    manifest = verify_files(directory, ["occupation_nodes.parquet", "occupation_embeddings.npy", "skill_to_occupation.npz",
                                        "occupation_to_skill.npz", "encoder_config.json", "summary.json"])
    if digest(root / candidate_dir / "manifest.json") != manifest["candidate_manifest_sha256"]:
        raise ValueError("Occupation and candidate versions differ")
    nodes = read_frame(directory / "occupation_nodes.parquet")
    x = np.load(io.BytesIO((directory / "occupation_embeddings.npy").read_bytes()), allow_pickle=False)
    forward, reverse = [sparse.load_npz(io.BytesIO((directory / f"{name}.npz").read_bytes()))
                        for name in ["skill_to_occupation", "occupation_to_skill"]]
    n = manifest["summary"]["candidate_count"]
    if (forward.shape != (len(nodes), n) or reverse.shape != (n, len(nodes)) or x.ndim != 2 or x.shape[0] != len(nodes)
            or not np.array_equal(nodes.occupation_id, np.arange(len(nodes))) or not np.isfinite(x).all()
            or not np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-5)):
        raise ValueError("Invalid occupation bundle dimensions or features")
    for matrix in [forward, reverse]:
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        if not np.isfinite(matrix.data).all() or (matrix.data <= 0).any() or not np.all(np.isclose(totals, 1.) | (totals == 0)):
            raise ValueError("Invalid normalized occupation relations")
    if (forward.astype(bool) != reverse.T.astype(bool)).nnz:
        raise ValueError("Occupation relation directions disagree")
    return {"features": x, "skill_to_occupation": forward, "occupation_to_skill": reverse,
            "available": np.diff(reverse.indptr) > 0}
