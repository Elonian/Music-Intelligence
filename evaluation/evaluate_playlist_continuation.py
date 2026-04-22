#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_playlist_continuation.audio_similarity import get_embeddings, make_audio_rankings  # noqa: E402
from scripts.automatic_playlist_continuation.collaborative_filtering import make_cf_rankings, train_wrmf  # noqa: E402
from scripts.automatic_playlist_continuation.data import build_interaction_samples, split_query_targets  # noqa: E402
from scripts.automatic_playlist_continuation.metrics import evaluate_rankings, get_mrr, get_precision  # noqa: E402
from utils.io_helpers import ensure_dir, save_json, write_csv_rows  # noqa: E402
from utils.playlist_continuation import (  # noqa: E402
    ensure_embeddings_extracted,
    load_playlists,
    playlist_tids,
    summarize_playlist_collection,
)
from utils.project_paths import (  # noqa: E402
    AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_DIR,
    AUTOMATIC_PLAYLIST_CONTINUATION_METRICS_DIR,
    AUTOMATIC_PLAYLIST_CONTINUATION_RANKING_DIR,
    AUTOMATIC_PLAYLIST_CONTINUATION_TEST_JSON,
    AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON,
)


def _select_playlists(playlists: dict[str, list[dict[str, str]]], max_playlists: int | None) -> dict[str, list[dict[str, str]]]:
    if max_playlists is None:
        return playlists
    selected_keys = sorted(playlists, key=lambda item: int(item))[:max_playlists]
    return {key: playlists[key] for key in selected_keys}


def _popularity(playlists: dict[str, list[dict[str, str]]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for playlist in playlists.values():
        counts.update(playlist_tids(playlist))
    return dict(counts)


def _query_target_coverage(test_playlists: dict[str, list[dict[str, str]]], known_tids: set[str]) -> dict:
    query_missing = 0
    target_missing = 0
    query_total = 0
    target_total = 0
    playlists_with_query_missing = 0
    playlists_with_target_missing = 0
    for playlist in test_playlists.values():
        query, targets = split_query_targets(playlist, query_size=2)
        query_total += len(query)
        target_total += len(targets)
        local_query_missing = sum(1 for tid in query if tid not in known_tids)
        local_target_missing = sum(1 for tid in targets if tid not in known_tids)
        query_missing += local_query_missing
        target_missing += local_target_missing
        playlists_with_query_missing += int(local_query_missing > 0)
        playlists_with_target_missing += int(local_target_missing > 0)
    return {
        "query_total": query_total,
        "query_missing": query_missing,
        "query_known_rate": 1.0 - query_missing / max(query_total, 1),
        "target_total": target_total,
        "target_missing": target_missing,
        "target_known_rate": 1.0 - target_missing / max(target_total, 1),
        "playlists_with_query_missing": playlists_with_query_missing,
        "playlists_with_target_missing": playlists_with_target_missing,
    }


def _ranking_preview(rankings: dict[str, list[str]], targets: dict[str, list[str]], limit: int = 10) -> list[dict]:
    rows = []
    for playlist_id in sorted(rankings, key=lambda item: int(item))[:limit]:
        rows.append(
            {
                "playlist_id": playlist_id,
                "top_10": rankings[playlist_id][:10],
                "targets": targets.get(playlist_id, []),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate automatic playlist continuation recommenders.")
    parser.add_argument("--train-json", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON)
    parser.add_argument("--test-json", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_TEST_JSON)
    parser.add_argument("--embedding-dir", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_METRICS_DIR)
    parser.add_argument("--ranking-dir", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_RANKING_DIR)
    parser.add_argument("--max-train-playlists", type=int, default=None)
    parser.add_argument("--max-test-playlists", type=int, default=None)
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--extract-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ranking_dir = ensure_dir(args.ranking_dir)
    train_playlists = _select_playlists(load_playlists(args.train_json), args.max_train_playlists)
    test_playlists = _select_playlists(load_playlists(args.test_json), args.max_test_playlists)
    data, tid_to_idx, idx_to_tid, tid_to_meta = build_interaction_samples(train_playlists, random_seed=args.seed)
    popularity = _popularity(train_playlists)
    train_known_tids = set(tid_to_idx)

    train_result = train_wrmf(
        data,
        num_users=len(train_playlists),
        num_items=len(tid_to_idx),
        num_factors=args.factors,
        alpha=args.alpha,
        lambda_reg=args.lambda_reg,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    cf_rankings, cf_targets = make_cf_rankings(
        train_result.model,
        test_playlists,
        train_result.model.item_factors.weight,
        idx_to_tid,
        tid_to_idx,
    )
    cf_metrics = evaluate_rankings(
        cf_rankings,
        cf_targets,
        catalog_size=len(tid_to_idx),
        popularity=popularity,
    )
    cf_metrics["assignment_mrr"] = get_mrr(cf_rankings, cf_targets)
    cf_metrics["assignment_precision_at_10"] = get_precision(cf_rankings, cf_targets, k=10)
    save_json(output_dir / "collaborative_filtering_metrics.json", cf_metrics)
    write_csv_rows(ranking_dir / "collaborative_filtering_preview.csv", _ranking_preview(cf_rankings, cf_targets, limit=len(test_playlists)))

    audio_metrics = None
    if not args.skip_audio:
        embedding_dir = args.embedding_dir
        if args.extract_embeddings or not embedding_dir.exists():
            embedding_dir = ensure_embeddings_extracted(args.embedding_dir)
        tids = list(tid_to_meta.keys())
        embedding_matrix, loaded_tids = get_embeddings(embedding_dir, tids)
        audio_rankings, audio_targets = make_audio_rankings(
            embedding_dir,
            test_playlists,
            embedding_matrix,
            tid_to_idx,
            loaded_tids,
        )
        audio_metrics = evaluate_rankings(
            audio_rankings,
            audio_targets,
            catalog_size=len(loaded_tids),
            popularity=popularity,
        )
        audio_metrics["assignment_mrr"] = get_mrr(audio_rankings, audio_targets)
        audio_metrics["assignment_precision_at_10"] = get_precision(audio_rankings, audio_targets, k=10)
        audio_metrics["loaded_embeddings"] = len(loaded_tids)
        save_json(output_dir / "audio_similarity_metrics.json", audio_metrics)
        write_csv_rows(ranking_dir / "audio_similarity_preview.csv", _ranking_preview(audio_rankings, audio_targets, limit=len(test_playlists)))

    summary = {
        "train": summarize_playlist_collection(train_playlists),
        "test": summarize_playlist_collection(test_playlists),
        "interaction_rows": len(data),
        "positive_rows": int(sum(row[2] for row in data)),
        "negative_rows": int(sum(1 for row in data if row[2] == 0)),
        "query_target_train_coverage": _query_target_coverage(test_playlists, train_known_tids),
        "wrmf_training": train_result.summary(),
        "collaborative_filtering": {
            key: value for key, value in cf_metrics.items() if key != "per_playlist"
        },
        "audio_similarity": None if audio_metrics is None else {key: value for key, value in audio_metrics.items() if key != "per_playlist"},
    }
    save_json(output_dir / "playlist_continuation_summary.json", summary)
    print(output_dir / "playlist_continuation_summary.json")


if __name__ == "__main__":
    main()
