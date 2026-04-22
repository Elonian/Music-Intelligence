from __future__ import annotations

import numpy as np

from scripts.automatic_playlist_continuation.data import split_query_targets
from utils.playlist_continuation import PlaylistDict, load_track_embedding, resolve_embedding_dir


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)
    query_norm = max(float(np.linalg.norm(query)), 1e-12)
    matrix_norm = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    return matrix @ query / (matrix_norm * query_norm)


def get_average_audio_embedding(playlist: list[str], embedding_directory) -> np.ndarray:
    embeddings = []
    for tid in playlist:
        embedding = load_track_embedding(tid, embedding_directory, allow_missing=True)
        if embedding is not None:
            embeddings.append(embedding)
    if not embeddings:
        raise ValueError("Playlist has no loadable audio embeddings.")
    return np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32, copy=False)


def get_embeddings(embedding_directory, tids: list[str]) -> tuple[np.ndarray, list[str]]:
    directory = resolve_embedding_dir(embedding_directory)
    rows: list[np.ndarray] = []
    loaded_tids: list[str] = []
    for tid in tids:
        embedding = load_track_embedding(tid, directory, allow_missing=True)
        if embedding is None:
            continue
        rows.append(embedding)
        loaded_tids.append(tid)
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(rows, axis=0).astype(np.float32, copy=False), loaded_tids


def get_similarity(
    playlist: list[str],
    playlist_embedding: np.ndarray,
    embedding_matrix: np.ndarray,
    tid_to_idx: dict[str, int],
    tids: list[str],
    n: int | None = None,
) -> tuple[list[float], list[str]]:
    del tid_to_idx
    if embedding_matrix.size == 0:
        return [], []
    similarities = _cosine_scores(playlist_embedding, embedding_matrix)
    playlist_set = set(playlist)
    sorted_indices = np.argsort(-similarities)
    filtered = [int(index) for index in sorted_indices if tids[int(index)] not in playlist_set]
    if n is not None:
        filtered = filtered[:n]
    return [float(similarities[index]) for index in filtered], [tids[index] for index in filtered]


def make_audio_rankings(
    directory,
    test_playlists: PlaylistDict,
    embedding_matrix: np.ndarray,
    tid_to_idx: dict[str, int],
    tids: list[str],
    query_size: int = 2,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rankings: dict[str, list[str]] = {}
    targets: dict[str, list[str]] = {}
    known_tids = set(tids)
    for playlist_id, playlist in test_playlists.items():
        query, target = split_query_targets(playlist, query_size=query_size, known_tids=known_tids)
        target = [tid for tid in target if tid in tid_to_idx or tid in known_tids]
        targets[playlist_id] = target
        if not query:
            rankings[playlist_id] = []
            continue
        playlist_embedding = get_average_audio_embedding(query, directory)
        _scores, ranking = get_similarity(
            query,
            playlist_embedding,
            embedding_matrix,
            tid_to_idx,
            tids,
            n=None,
        )
        rankings[playlist_id] = ranking
    return rankings, targets
