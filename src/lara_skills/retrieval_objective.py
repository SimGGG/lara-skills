"""Positive-supervised retrieval with a soft text prior for incomplete judgments."""

import math

import torch
from torch.nn import functional as F

from .ranking import pairwise_ranking_loss


def partial_retrieval_targets(cosine, query_ids, skill_ids, grades, temperature, positive_mass):
    """Return a distribution, NOT invented binary labels for unknown skills.

    Explicit zeros receive no target mass. Unknowns retain soft semantic mass.
    The positive mixture also reduces unknowns' relative mass; it is a modeling
    assumption, not an unbiased estimator of complete relevance judgments.
    Queries without positives retain the conditioned semantic distribution.
    """
    if (not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(positive_mass) or not 0 < positive_mass < 1):
        raise ValueError("Invalid retrieval target temperature or positive mass")
    if cosine.ndim != 2 or not torch.isfinite(cosine).all():
        raise ValueError("Expected finite all-candidate cosine scores")
    if not (query_ids.ndim == skill_ids.ndim == grades.ndim == 1
            and len(query_ids) == len(skill_ids) == len(grades)):
        raise ValueError("Invalid judgment arrays")
    if (not torch.isin(grades, grades.new_tensor([0, 1, 2])).all()
            or (query_ids < 0).any() or (query_ids >= len(cosine)).any()
            or (skill_ids < 0).any() or (skill_ids >= cosine.shape[1]).any()):
        raise ValueError("Invalid judgment IDs or grades")
    flat = query_ids * cosine.shape[1] + skill_ids
    if len(torch.unique(flat)) != len(flat):
        raise ValueError("Duplicate judgments")
    with torch.no_grad():
        logits = cosine.detach().clone() / temperature
        zeros = grades == 0
        logits[query_ids[zeros], skill_ids[zeros]] = -torch.inf
        if torch.isneginf(logits).all(1).any():
            raise ValueError("A query has no candidate with nonzero target mass")
        teacher = torch.softmax(logits, dim=1)
        positives = torch.zeros_like(teacher)
        mask = grades > 0
        positives[query_ids[mask], skill_ids[mask]] = 1.
        counts = positives.sum(1, keepdim=True)
        mix = (counts > 0).to(cosine.dtype) * positive_mass
        target = (1 - mix) * teacher + mix * positives / counts.clamp_min(1)
        if not torch.isfinite(target).all() or not torch.allclose(target.sum(1), target.new_ones(len(target)), atol=1e-6):
            raise ValueError("Invalid soft retrieval target")
    return target


def task_balanced_weights(task_ids, dtype=torch.float32):
    if task_ids.ndim != 1 or not len(task_ids):
        raise ValueError("Expected nonempty work type IDs")
    groups, inverse, counts = torch.unique(task_ids, return_inverse=True, return_counts=True)
    return 1. / (len(groups) * counts[inverse].to(dtype))


def retrieval_training_loss(scores, cosine, target, task_ids, query_ids, skill_ids,
                            high, low, pair_weights, pair_tasks, config):
    if scores.shape != target.shape or scores.shape != cosine.shape or not torch.isfinite(scores).all():
        raise ValueError("Invalid full-dictionary training scores")
    temperature = config["retrieval_temperature"]
    logp = F.log_softmax(scores / temperature, dim=1)
    per_query = F.kl_div(logp, target, reduction="none").sum(1)
    weights = task_balanced_weights(task_ids, scores.dtype)
    retrieval = (per_query * weights).sum()
    if pair_tasks:
        pair = pairwise_ranking_loss(scores[query_ids, skill_ids], high, low, pair_weights,
                                     pair_tasks, config["rank_temperature"])
    else:
        pair = scores.sum() * 0
    residual = (((scores - cosine) / config["residual_bound"]).square().mean(1) * weights).sum()
    total = retrieval + config["auxiliary_rank_weight"] * pair + config["residual_regularization"] * residual
    return total, {"retrieval_kl": retrieval.detach(), "auxiliary_pairwise": pair.detach(),
                   "residual_penalty": residual.detach()}


def retrieval_selection_key(summary):
    """Call in epoch/alpha ascending order; strict comparison retains first tie."""
    row = summary[(summary.positive_grade_min == 1) & (summary.k == 5)]
    if len(row) != 1:
        raise ValueError("Missing unique primary retrieval result")
    key = (float(row.iloc[0].recall_macro), float(row.iloc[0].hit_macro))
    if not all(math.isfinite(value) for value in key):
        raise ValueError("No eligible known-positive validation queries")
    return key
