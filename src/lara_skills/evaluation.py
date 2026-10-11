"""Explicitly judged-pool metrics; unknown labels are never filled with zero."""

import numpy as np
import pandas as pd


def ranking_pairs(labels):
    high, low, weights = [], [], []
    tasks = 0
    for _, group in labels.groupby("query_id", sort=True):
        ids = group.index.to_numpy()
        grades = group.relevance_score.to_numpy()
        a, b = np.where(grades[:, None] > grades[None, :])
        if len(a):
            tasks += 1
            high.extend(ids[a])
            low.extend(ids[b])
            weights.extend((grades[a] - grades[b]) / len(a))
    if not tasks:
        raise ValueError("No unequal-grade pairs in the training labels")
    return np.array(high), np.array(low), np.array(weights, dtype=np.float32), tasks


def judged_metrics(labels, scores, k=5):
    if k < 1 or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid metric inputs")
    work = labels.copy()
    work["score"] = scores
    rows = []
    for query_id, group in work.groupby("query_id", sort=True):
        # A stable skill-ID tie rule is independent of relevance grades.
        group = group.sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
        grades = group.relevance_score.to_numpy(dtype=float)
        n = min(k, len(grades))
        discounts = np.log2(np.arange(n) + 2)
        dcg = float(((2 ** grades[:n] - 1) / discounts).sum())
        ideal = float(((2 ** np.sort(grades)[::-1][:n] - 1) / discounts).sum())
        original = work[work.query_id == query_id]
        y, s = original.relevance_score.to_numpy(), original.score.to_numpy()
        a, b = np.where(y[:, None] > y[None, :])
        accuracy = float(((s[a] > s[b]) + .5 * (s[a] == s[b])).mean()) if len(a) else None
        rows.append({"query_id": int(query_id), "sample_id": group.iloc[0].sample_id,
                     "perspective": group.iloc[0].perspective, "judged_count": len(grades),
                     "judged_ndcg_at_5": dcg / ideal if ideal > 0 else None,
                     "judged_precision_at_min5": float((grades[:n] > 0).mean()),
                     "pairwise_accuracy": accuracy, "unequal_grade_pairs": len(a)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No judged queries to evaluate")
    by_perspective = {}
    fields = ["judged_ndcg_at_5", "judged_precision_at_min5", "pairwise_accuracy"]
    for perspective, group in detail.groupby("perspective"):
        by_perspective[perspective] = {field: float(group[field].dropna().mean()) if group[field].notna().any() else None for field in fields}
        by_perspective[perspective].update({"queries": len(group), "ndcg_eligible_queries": int(group.judged_ndcg_at_5.notna().sum()),
                                          "pairwise_eligible_queries": int(group.pairwise_accuracy.notna().sum())})
    macro = {field: float(np.mean([v[field] for v in by_perspective.values() if v[field] is not None]))
             if any(v[field] is not None for v in by_perspective.values()) else None for field in fields}
    return {"scope": "ranking_within_human_judged_candidates_only", "k": k,
            "queries": len(detail), "queries_with_at_least_k_judged": int((detail.judged_count >= k).sum()),
            "perspective_macro": macro, "by_perspective": by_perspective}, detail


def full_ranking_coverage(labels, full_scores, k=5):
    """Report knownness of actual top-k, not unjudged-as-negative precision."""
    records = []
    for query_id, group in labels.groupby("query_id", sort=True):
        selected = np.argsort(-full_scores[int(query_id)], kind="stable")[:k]
        lookup = dict(zip(group.candidate_id, group.relevance_score))
        judged = [lookup[int(c)] for c in selected if int(c) in lookup]
        records.append({"query_id": int(query_id), "returned": len(selected), "judged": len(judged),
                        "known_relevant": sum(v > 0 for v in judged), "unjudged": len(selected) - len(judged)})
    total = sum(r["returned"] for r in records)
    known = sum(r["judged"] for r in records)
    relevant = sum(r["known_relevant"] for r in records)
    return {"top_k_slots": total, "judged_slots": known, "unjudged_slots": total - known,
            "judgment_coverage": known / total if total else None,
            "relevant_fraction_among_judged_slots": relevant / known if known else None,
            "full_universe_precision_or_recall_estimated": False}
