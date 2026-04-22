from __future__ import annotations

import math
from statistics import median

import numpy as np


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _target_set(targets: list[str]) -> set[str]:
    return set(targets)


def reciprocal_ranks(ranking: list[str], targets: list[str]) -> list[float]:
    ranks = {item: index + 1 for index, item in enumerate(_dedupe(ranking))}
    return [1.0 / ranks[target] if target in ranks else 0.0 for target in _target_set(targets)]


def playlist_mrr(ranking: list[str], targets: list[str]) -> float:
    values = reciprocal_ranks(ranking, targets)
    return float(np.mean(values)) if values else 0.0


def precision_at_k(ranking: list[str], targets: list[str], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    hits = len(set(_dedupe(ranking)[:k]) & _target_set(targets))
    return float(hits / k)


def target_normalized_precision_at_k(ranking: list[str], targets: list[str], k: int = 10) -> float:
    target_items = _target_set(targets)
    if not target_items:
        return 0.0
    hits = len(set(_dedupe(ranking)[:k]) & target_items)
    return float(hits / len(target_items))


def recall_at_k(ranking: list[str], targets: list[str], k: int = 10) -> float:
    return target_normalized_precision_at_k(ranking, targets, k=k)


def hit_rate_at_k(ranking: list[str], targets: list[str], k: int = 10) -> float:
    if not targets:
        return 0.0
    return float(len(set(_dedupe(ranking)[:k]) & _target_set(targets)) > 0)


def average_precision_at_k(ranking: list[str], targets: list[str], k: int = 10) -> float:
    target_items = _target_set(targets)
    if not target_items:
        return 0.0
    seen_hits = 0
    precision_sum = 0.0
    for index, item in enumerate(_dedupe(ranking)[:k], start=1):
        if item in target_items:
            seen_hits += 1
            precision_sum += seen_hits / index
    return float(precision_sum / min(len(target_items), k))


def ndcg_at_k(ranking: list[str], targets: list[str], k: int = 10) -> float:
    target_items = _target_set(targets)
    if not target_items:
        return 0.0
    dcg = 0.0
    for index, item in enumerate(_dedupe(ranking)[:k], start=1):
        if item in target_items:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(target_items), k)
    ideal = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return float(dcg / ideal) if ideal > 0 else 0.0


def first_relevant_rank(ranking: list[str], targets: list[str]) -> int | None:
    target_items = _target_set(targets)
    if not target_items:
        return None
    for index, item in enumerate(_dedupe(ranking), start=1):
        if item in target_items:
            return index
    return None


def get_mrr(rankings: dict[str, list[str]], targets: dict[str, list[str]]) -> float:
    values = [playlist_mrr(rankings.get(key, []), target_items) for key, target_items in targets.items()]
    return float(np.mean(values)) if values else 0.0


def get_precision(rankings: dict[str, list[str]], targets: dict[str, list[str]], k: int = 10) -> float:
    values = [target_normalized_precision_at_k(rankings.get(key, []), target_items, k=k) for key, target_items in targets.items()]
    return float(np.mean(values)) if values else 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]],
    targets: dict[str, list[str]],
    catalog_size: int | None = None,
    k_values: tuple[int, ...] = (5, 10, 20, 50, 100),
    popularity: dict[str, int] | None = None,
) -> dict:
    rows = []
    first_ranks: list[int] = []
    recommended_all: set[str] = set()
    for playlist_id, target_items in targets.items():
        ranking = _dedupe(rankings.get(playlist_id, []))
        target_items = _dedupe(target_items)
        recommended_all.update(ranking[: max(k_values)])
        first_rank = first_relevant_rank(ranking, target_items)
        if first_rank is not None:
            first_ranks.append(first_rank)
        row = {
            "playlist_id": playlist_id,
            "target_count": len(target_items),
            "ranking_count": len(ranking),
            "mrr": playlist_mrr(ranking, target_items),
            "first_relevant_rank": first_rank,
        }
        for k in k_values:
            row[f"precision_at_{k}"] = precision_at_k(ranking, target_items, k=k)
            row[f"target_precision_at_{k}"] = target_normalized_precision_at_k(ranking, target_items, k=k)
            row[f"recall_at_{k}"] = recall_at_k(ranking, target_items, k=k)
            row[f"hit_rate_at_{k}"] = hit_rate_at_k(ranking, target_items, k=k)
            row[f"map_at_{k}"] = average_precision_at_k(ranking, target_items, k=k)
            row[f"ndcg_at_{k}"] = ndcg_at_k(ranking, target_items, k=k)
        rows.append(row)

    summary: dict[str, float | int | None | list[dict]] = {
        "playlists": len(rows),
        "mrr": float(np.mean([row["mrr"] for row in rows])) if rows else 0.0,
        "median_first_relevant_rank": float(median(first_ranks)) if first_ranks else None,
        "mean_first_relevant_rank": float(np.mean(first_ranks)) if first_ranks else None,
        "missing_first_relevant_rate": float(1.0 - len(first_ranks) / len(rows)) if rows else 0.0,
        "catalog_coverage_at_max_k": float(len(recommended_all) / catalog_size) if catalog_size else None,
    }
    for k in k_values:
        for metric in ("precision", "target_precision", "recall", "hit_rate", "map", "ndcg"):
            key = f"{metric}_at_{k}"
            summary[key] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    if popularity:
        max_popularity = max(popularity.values()) if popularity else 1
        novelty_values = []
        for row_key, ranking in rankings.items():
            del row_key
            top_items = _dedupe(ranking)[:10]
            if top_items:
                novelty_values.append(float(np.mean([1.0 - popularity.get(item, 0) / max_popularity for item in top_items])))
        summary["mean_novelty_at_10"] = float(np.mean(novelty_values)) if novelty_values else 0.0
    summary["per_playlist"] = rows
    return summary
