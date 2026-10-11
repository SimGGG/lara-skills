"""Full-dictionary training of LARA / LARA-NA with validation-only checkpoint selection."""

import gc
import hashlib
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .evaluation import judged_metrics, ranking_pairs
from .hierarchy import load_hierarchy
from .models import MODEL_NAMES, make_ranker
from .occupations import load_occupations
from .pipeline_io import digest, output_run, write_json, write_parquet
from .ranking import baseline_scores, torch_adjacency
from .recommender_data import load_development_split, load_market
from .retrieval_evaluation import retrieval_metrics
from .retrieval_objective import partial_retrieval_targets, retrieval_selection_key, retrieval_training_loss

BASELINES = {"popularity", "semantic", "diffusion"}


def _validate_config(config):
    if config["score_mode"] != "bounded_residual" or config["top_k"] != 5:
        raise ValueError("Retrieval training requires bounded residual scores and top_k=5")
    for key in ["retrieval_temperature", "rank_temperature", "residual_bound", "learning_rate", "gradient_clip"]:
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid positive setting: {key}")
    for key in ["auxiliary_rank_weight", "residual_regularization"]:
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid nonnegative setting: {key}")
    if not 0 < config["positive_target_mass"] < 1:
        raise ValueError("positive_target_mass must be between zero and one")
    models = config["training_models"]
    if not models or len(set(models)) != len(models) or not set(models) <= MODEL_NAMES:
        raise ValueError(f"training_models must be a unique subset of {sorted(MODEL_NAMES)}")
    if config["default_engine"] not in set(models) | BASELINES:
        raise ValueError("default_engine must be a trained model or a baseline")
    if not isinstance(config["epochs"], int) or config["epochs"] < 1:
        raise ValueError("epochs must be positive")
    alphas = config["diffusion_alphas"]
    if not alphas or any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas):
        raise ValueError("Invalid diffusion alpha grid")


def fit_retrieval_recommenders(root, config):
    _validate_config(config)
    root = Path(root)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select device cpu")
    candidates, x_np, adj_np, freq_np = load_market(root, config["candidate_dir"])
    hierarchy = load_hierarchy(root, config["hierarchy_dir"], config["candidate_dir"])
    occupations = load_occupations(root, config["occupation_dir"], config["candidate_dir"])
    train = load_development_split(root, config, "train", candidates)
    valid = load_development_split(root, config, "validation", candidates)
    if set(train["queries"].sample_id) & set(valid["queries"].sample_id):
        raise ValueError("Patent leakage")
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
    x, adjacency, freq = tensor(x_np), torch_adjacency(adj_np, device), tensor(freq_np)
    for data in [train, valid]:
        data["q"] = tensor(data["features"])
        data["p"] = tensor(data["queries"].perspective_id.to_numpy(), torch.long)
        data["qi"] = tensor(data["labels"].query_id.to_numpy(), torch.long)
        data["si"] = tensor(data["labels"].candidate_id.to_numpy(), torch.long)
        data["cosine"] = data["q"] @ x.T
    if any(g.relevance_score.nunique() > 1 for _, g in train["labels"].groupby("query_id")):
        hi, lo, pw, pair_tasks = ranking_pairs(train["labels"])
    else:
        hi, lo, pw, pair_tasks = [], [], [], 0
    hi, lo, pw = tensor(hi, torch.long), tensor(lo, torch.long), tensor(pw)
    target = partial_retrieval_targets(train["cosine"], train["qi"], train["si"],
                                      tensor(train["labels"].relevance_score.to_numpy(), torch.long),
                                      config["retrieval_temperature"], config["positive_target_mass"])
    encoder = json.loads((root / config["patent_embedding_dir"] / "encoder_config.json").read_text(encoding="utf-8"))
    ks = sorted(set([5] + config.get("retrieval_ks", [5, 10, 20, 50, 100])))
    if max(ks) > len(candidates):
        raise ValueError("Retrieval K exceeds dictionary size")
    input_keys = ["candidate_dir", "patent_embedding_dir", "split_dir", "hierarchy_dir", "occupation_dir"]
    input_hashes = {key: digest(root / config[key] / "manifest.json") for key in input_keys}
    comparisons, retrieval_rows = [], []
    with output_run(root, root / config["output_dir"], "retrieval_training", config) as (out, manifest):
        write_json(out / "config.json", config)
        write_json(out / "encoder_config.json", encoder)
        target_mass = target[train["qi"], train["si"]].cpu().numpy()
        target_audit = train["labels"][["query_id", "skill_code", "relevance_score"]].copy()
        target_audit["soft_target_mass"] = target_mass
        write_parquet(target_audit, out / "training_target_judgments.parquet")

        def assess(scores, data=valid):
            return retrieval_metrics(scores, data["queries"], data["labels"], ks)

        def record(name, scores, metadata):
            summary, detail, ranks, top = assess(scores)
            key = retrieval_selection_key(summary)
            judged, _ = judged_metrics(valid["labels"], scores[valid["qi"].cpu(), valid["si"].cpu()])
            comparison = {"model": name, "known_recall_at_5": key[0], "known_hit_at_5": key[1],
                          **judged["perspective_macro"], "unique_top5_skills": len(np.unique(top[:, :5])), **metadata}
            comparisons.append(comparison)
            summary.insert(0, "model", name)
            retrieval_rows.append(summary)
            summary.to_csv(out / f"{name}_retrieval.csv", index=False)
            write_parquet(detail, out / f"{name}_retrieval_queries.parquet")
            write_parquet(ranks, out / f"{name}_known_label_ranks.parquet")
            write_json(out / f"{name}_validation.json", {"retrieval": summary.to_dict(orient="records"),
                                                       "judged_secondary": judged, "metadata": metadata})
            labels = valid["labels"][["sample_id", "perspective", "skill_code", "relevance_score"]].copy()
            labels["score"] = scores[valid["qi"].cpu(), valid["si"].cpu()]
            write_parquet(labels, out / f"{name}_judged_scores.parquet")
            rows = []
            for qid, query in valid["queries"].iterrows():
                for rank, cid in enumerate(top[qid], 1):
                    c = candidates.iloc[cid]
                    rows.append({"sample_id": query.sample_id, "perspective": query.perspective, "rank": rank,
                                 "skill_code": c.skill_code, "skill_name": c.skill_name, "score": float(scores[qid, cid])})
            frame = pd.DataFrame(rows)
            write_parquet(frame[frame["rank"] <= 5], out / f"{name}_validation_top5.parquet")
            write_parquet(frame, out / f"{name}_validation_top100.parquet")
            np.save(out / f"{name}_validation_all_scores.npy", scores, allow_pickle=False)
            print(json.dumps(comparison), flush=True)

        for name in ["popularity", "semantic"]:
            scores = baseline_scores(name, valid["features"], x_np, candidates.posting_count.to_numpy(), adj_np)
            record(name, scores, {"best_epoch": None})
        trials, best_key, best_alpha = [], (-math.inf, -math.inf), None
        for alpha in sorted(set(config["diffusion_alphas"])):
            scores = baseline_scores("diffusion", valid["features"], x_np, candidates.posting_count.to_numpy(), adj_np, alpha)
            key = retrieval_selection_key(assess(scores)[0])
            trials.append({"alpha": alpha, "known_recall_at_5": key[0], "known_hit_at_5": key[1]})
            if key > best_key:
                best_key, best_alpha = key, alpha
        write_json(out / "diffusion_selection.json", {"alpha": best_alpha, "trials": trials,
                                                     "selection_split": "validation", "selection": "recall5_hit5_smaller_alpha"})
        record("diffusion", baseline_scores("diffusion", valid["features"], x_np, candidates.posting_count.to_numpy(), adj_np, best_alpha),
               {"alpha": best_alpha, "best_epoch": None})

        for name in config["training_models"]:
            random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
            model = make_ranker(name, x.shape[1], len(encoder["perspectives"]), config, hierarchy, occupations).to(device)
            initial = hashlib.sha256()
            for k, v in sorted(model.state_dict().items()):
                initial.update(k.encode()); initial.update(v.detach().cpu().contiguous().numpy().tobytes())
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            history, best_key, best_epoch, best_state = [], (-math.inf, -math.inf), 0, None
            for epoch in range(config["epochs"] + 1):
                components, loss_value, grad_value = {}, None, None
                if epoch:
                    model.train(); optimizer.zero_grad(set_to_none=True)
                    predictions = model.score_all(train["q"], train["p"], x, adjacency, freq, config["score_chunk_size"])
                    loss, parts = retrieval_training_loss(predictions, train["cosine"], target, train["p"],
                                                          train["qi"], train["si"], hi, lo, pw, pair_tasks, config)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite retrieval loss")
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    loss_value, grad_value = float(loss.detach()), float(grad)
                    components = {k: float(v) for k, v in parts.items()}
                    del predictions, loss, parts
                model.eval()
                with torch.no_grad():
                    scores = model.score_all(valid["q"], valid["p"], x, adjacency, freq, config["score_chunk_size"]).cpu().numpy()
                if epoch == 0 and not np.allclose(scores, valid["cosine"].cpu().numpy(), atol=1e-6, rtol=1e-5):
                    raise ValueError("Epoch zero must reproduce the semantic baseline")
                key = retrieval_selection_key(assess(scores)[0])
                judged, _ = judged_metrics(valid["labels"], scores[valid["qi"].cpu(), valid["si"].cpu()])
                history.append({"epoch": epoch, "training_loss": loss_value, "gradient_norm": grad_value, **components,
                                "validation_known_recall_at_5": key[0], "validation_known_hit_at_5": key[1],
                                "validation_judged_ndcg_at_5": judged["perspective_macro"]["judged_ndcg_at_5"]})
                if key > best_key:
                    best_key, best_epoch = key, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if epoch <= 1 or epoch % 10 == 0:
                    print(json.dumps({"model": name, **history[-1]}), flush=True)
            model.load_state_dict(best_state); model.eval()
            buffer = io.BytesIO()
            torch.save({"state_dict": best_state, "model": name, "input_dim": x.shape[1],
                        "perspective_count": len(encoder["perspectives"]), "config": config,
                        "best_epoch": best_epoch, "selection_split": "validation",
                        "selection_metric": "known_recall5_then_hit5_then_earliest_epoch_including_zero",
                        "initial_state_sha256": initial.hexdigest()}, buffer)
            (out / f"{name}.pt").write_bytes(buffer.getvalue())
            pd.DataFrame(history).to_csv(out / f"{name}_history.csv", index=False)
            with torch.no_grad():
                scores = model.score_all(valid["q"], valid["p"], x, adjacency, freq, config["score_chunk_size"]).cpu().numpy()
                train_scores = model.score_all(train["q"], train["p"], x, adjacency, freq, config["score_chunk_size"]).cpu().numpy()
            record(name, scores, {"best_epoch": best_epoch, "seed": config["seed"], "initial_state_sha256": initial.hexdigest()})
            train_summary = assess(train_scores, train)[0]
            train_summary.to_csv(out / f"{name}_train_retrieval.csv", index=False)
            np.save(out / f"{name}_train_all_scores.npy", train_scores, allow_pickle=False)
            delta = scores - valid["cosine"].cpu().numpy()
            write_json(out / f"{name}_score_diagnostics.json", {
                "max_absolute_residual": float(np.abs(delta).max()), "residual_bound": config["residual_bound"],
                "mean_within_query_cosine_std": float(valid["cosine"].std(1, correction=0).mean()),
                "mean_within_query_residual_std": float(delta.std(1).mean()),
                "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
                "network_coefficient": float(torch.tanh(model.network_scale_raw).detach())})
            with torch.no_grad():
                market = model.encode_market(x, adjacency)
                query = model.encode_queries(valid["q"], valid["p"], x, market)
                diagnostics = pd.DataFrame(model.attention_diagnostics(query))
            write_parquet(diagnostics, out / f"{name}_validation_node_attention.parquet")
            del market, query, model, optimizer, best_state
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        pd.DataFrame(comparisons).to_csv(out / "validation_comparison.csv", index=False, encoding="utf-8-sig")
        pd.concat(retrieval_rows).to_csv(out / "retrieval_comparison.csv", index=False, encoding="utf-8-sig")
        summary = {"stage": "single_seed_retrieval_training", "models": comparisons,
                   "selection_metric": "known_recall5_then_hit5_then_earliest_epoch_including_zero",
                   "training_patents": int(train["queries"].sample_id.nunique()),
                   "validation_patents": int(valid["queries"].sample_id.nunique()),
                   "test_labels_read": False, "test_model_evaluation_executed": False,
                   "unknowns_are_binary_negatives": False, "soft_targets_are_human_labels": False,
                   "teacher": "frozen_text_cosine_distribution_conditioned_on_not_explicit_zero",
                   "training_objective": config["training_objective"], "default_engine": config["default_engine"],
                   "candidate_count": len(candidates)}
        write_json(out / "summary.json", summary)
        if any(digest(root / config[key] / "manifest.json") != value for key, value in input_hashes.items()):
            raise ValueError("Input manifests changed during training")
        manifest.update({"summary": summary, "input_manifests": input_hashes,
                         "torch_version": str(torch.__version__), "cuda": torch.version.cuda,
                         "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    return summary
