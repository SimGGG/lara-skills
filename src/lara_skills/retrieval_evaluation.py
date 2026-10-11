"""Full-dictionary retrieval of known positives, with unjudged items preserved."""

import numpy as np
import pandas as pd


def retrieval_metrics(scores, queries, labels, ks=(5, 10, 20, 50, 100)):
    """Recall/Hit of known grades >=1 and >=2 from the ENTIRE score matrix.

    Unknown items remain in the ranking, but never become negative judgments.
    Queries without known positives are undefined for Recall and Hit.
    """
    scores = np.asarray(scores)
    if (scores.ndim != 2 or scores.shape[0] != len(queries) or scores.shape[1] < 1
            or not np.isfinite(scores).all()):
        raise ValueError("Expected finite all-candidate scores for every input query")
    if not np.array_equal(queries.query_id.to_numpy(), np.arange(len(queries))):
        raise ValueError("Query rows must match score matrix rows")
    if queries.duplicated(["sample_id", "perspective"]).any():
        raise ValueError("Duplicate input queries")
    if (not labels.relevance_score.isin([0, 1, 2]).all()
            or labels.duplicated(["query_id", "candidate_id"]).any()
            or not labels.query_id.isin(queries.query_id).all()
            or not labels.candidate_id.isin(range(scores.shape[1])).all()):
        raise ValueError("Invalid or duplicate explicit judgments")
    if not ks or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 or k > scores.shape[1] for k in ks):
        raise ValueError("K must be a positive integer within the candidate universe")
    if len(set(ks)) != len(ks):
        raise ValueError("Duplicate cutoffs")
    order = np.argsort(-scores, axis=1, kind="stable")
    inverse = np.empty_like(order)
    np.put_along_axis(inverse, order, np.arange(1, scores.shape[1] + 1)[None, :], axis=1)
    groups = {int(q): g for q, g in labels.groupby("query_id")}
    rows, known_rows = [], []
    for qid, query in queries.iterrows():
        judged = groups.get(qid, labels.iloc[:0])
        ids = judged.candidate_id.to_numpy(dtype=int)
        grades = judged.relevance_score.to_numpy(dtype=int)
        for cid, grade in zip(ids, grades):
            known_rows.append({"query_id": qid, "sample_id": query.sample_id, "perspective": query.perspective,
                               "candidate_id": int(cid), "relevance_score": int(grade),
                               "full_universe_rank": int(inverse[qid, cid]), "score": float(scores[qid, cid])})
        for threshold in [1, 2]:
            positives = ids[grades >= threshold]
            for k in ks:
                retrieved = order[qid, :k]
                hits = int(np.isin(positives, retrieved).sum())
                selected_known = np.isin(ids, retrieved)
                rows.append({"query_id": qid, "sample_id": query.sample_id, "perspective": query.perspective,
                             "positive_grade_min": threshold, "k": k, "known_positives": len(positives),
                             "positive_hits": hits, "known_positive_recall": hits / len(positives) if len(positives) else None,
                             "known_positive_hit": float(hits > 0) if len(positives) else None,
                             "judged_retrieved": int(selected_known.sum()),
                             "known_grade0_retrieved": int((selected_known & (grades == 0)).sum()),
                             "unjudged_retrieved": k - int(selected_known.sum())})
    detail = pd.DataFrame(rows)
    summaries = []
    for (threshold, k), g in detail.groupby(["positive_grade_min", "k"], sort=True):
        eligible = g[g.known_positives > 0]
        by_task = eligible.groupby("perspective")[["known_positive_recall", "known_positive_hit"]].mean()
        summaries.append({"positive_grade_min": int(threshold), "k": int(k),
                          "recall_macro": float(by_task.known_positive_recall.mean()) if len(by_task) else None,
                          "hit_macro": float(by_task.known_positive_hit.mean()) if len(by_task) else None,
                          "eligible_queries": len(eligible), "excluded_no_known_positive_queries": len(g) - len(eligible),
                          "eligible_work_types": len(by_task), "queries_with_hit": int((eligible.positive_hits > 0).sum()),
                          "positive_hits": int(eligible.positive_hits.sum()), "known_positives": int(eligible.known_positives.sum()),
                          "recall_micro": float(eligible.positive_hits.sum() / eligible.known_positives.sum()) if len(eligible) else None,
                          "total_slots_all_queries": int(k * len(g)), "judged_slots_all_queries": int(g.judged_retrieved.sum()),
                          "known_grade0_slots": int(g.known_grade0_retrieved.sum()),
                          "unjudged_slots": int(g.unjudged_retrieved.sum()),
                          "judgment_coverage_all_queries": float(g.judged_retrieved.sum() / (k * len(g)))})
    return pd.DataFrame(summaries), detail, pd.DataFrame(known_rows), order[:, :max(ks)]

