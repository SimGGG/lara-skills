"""Full-dictionary retrieval evaluation of a saved engine on one labeled split."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import judged_metrics
from .pipeline_io import digest, output_run, write_json, write_parquet
from .recommender import RecommendationEngine
from .recommender_data import load_labeled_split
from .retrieval_evaluation import retrieval_metrics


def evaluate_split(root, config):
    """Score every candidate for the patents of ``config['split']`` and report Recall@K / Hit@K.

    Unjudged skills stay in the ranking without becoming negatives. The test split
    is intended for one final run after every modeling choice has been fixed.
    """
    root = Path(root)
    split = config["split"]
    if split == "test":
        print("Note: evaluate the test split once, after all modeling choices are fixed on validation.", flush=True)
    engine = RecommendationEngine(root, config["model_dir"], model=config.get("model"), device=config.get("device", "cpu"))
    data_config = {"patent_embedding_dir": engine.config["patent_embedding_dir"], "split_dir": engine.config["split_dir"],
                   "task_mode": engine.config.get("task_mode", "work_types")}
    data = load_labeled_split(root, data_config, split, engine.candidates, allow_test=True)
    ks = sorted(set(int(k) for k in config.get("ks", [5, 10, 20, 50, 100])))
    if max(ks) > len(engine.candidates):
        raise ValueError("Retrieval K exceeds the candidate dictionary size")
    scores = engine.score_embeddings(data["features"], data["queries"].perspective_id.to_numpy())
    summary, detail, ranks, top = retrieval_metrics(scores, data["queries"], data["labels"], ks)
    judged, judged_detail = judged_metrics(data["labels"], scores[data["labels"].query_id.to_numpy(),
                                                                    data["labels"].candidate_id.to_numpy()])
    rows = []
    for qid, query in data["queries"].iterrows():
        for rank, cid in enumerate(top[qid], 1):
            candidate = engine.candidates.iloc[int(cid)]
            rows.append({"sample_id": query.sample_id, "perspective": query.perspective, "rank": rank,
                         "skill_code": candidate.skill_code, "skill_name": candidate.skill_name, "score": float(scores[qid, cid])})
    primary = summary[(summary.positive_grade_min == 1) & (summary.k == 5)].iloc[0]
    result = {"split": split, "model": engine.model_name, "queries": int(len(data["queries"])),
              "candidates": int(len(engine.candidates)), "ks": ks,
              "known_positive_recall_at_5": float(primary.recall_macro), "known_positive_hit_at_5": float(primary.hit_macro),
              "retrieval": summary.to_dict(orient="records"), "judged_pool_secondary": judged,
              "unjudged_are_negative": False, "test_labels_read": split == "test"}
    with output_run(root, root / config["output_dir"], f"evaluate_{split}", config) as (out, manifest):
        summary.to_csv(out / "retrieval.csv", index=False)
        write_parquet(detail, out / "query_metrics.parquet")
        write_parquet(ranks, out / "known_label_ranks.parquet")
        write_parquet(judged_detail, out / "judged_pool_metrics.parquet")
        write_parquet(pd.DataFrame(rows), out / "top_candidates.parquet")
        np.save(out / "all_scores.npy", scores, allow_pickle=False)
        write_json(out / "summary.json", result)
        manifest.update({"summary": {k: result[k] for k in ["split", "model", "queries", "candidates",
                                                             "known_positive_recall_at_5", "known_positive_hit_at_5"]},
                         "model_manifest_sha256": digest(root / config["model_dir"] / "manifest.json")})
    print(json.dumps({"split": split, "model": engine.model_name,
                      "known_positive_recall_at_5": result["known_positive_recall_at_5"],
                      "known_positive_hit_at_5": result["known_positive_hit_at_5"]}), flush=True)
    return result
