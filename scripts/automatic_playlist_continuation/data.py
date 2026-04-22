from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.playlist_continuation import PlaylistDict, collect_playlist_metadata, collect_unique_tids, playlist_tids


@dataclass(frozen=True)
class InteractionMaps:
    tid_to_idx: dict[str, int]
    idx_to_tid: dict[int, str]
    tid_to_meta: dict[str, tuple[str, str]]
    playlist_id_to_idx: dict[str, int]


def _playlist_index_map(playlists: PlaylistDict) -> dict[str, int]:
    keys = list(playlists.keys())
    try:
        ordered = sorted(keys, key=lambda item: int(item))
    except ValueError:
        ordered = sorted(keys)
    return {playlist_id: index for index, playlist_id in enumerate(ordered)}


def build_track_maps(playlists: PlaylistDict) -> InteractionMaps:
    tids = collect_unique_tids(playlists)
    tid_to_idx = {tid: index for index, tid in enumerate(tids)}
    idx_to_tid = {index: tid for tid, index in tid_to_idx.items()}
    return InteractionMaps(
        tid_to_idx=tid_to_idx,
        idx_to_tid=idx_to_tid,
        tid_to_meta=collect_playlist_metadata(playlists),
        playlist_id_to_idx=_playlist_index_map(playlists),
    )


def build_interaction_samples(
    playlists: PlaylistDict,
    random_seed: int = 42,
    negative_ratio: float = 1.0,
) -> tuple[list[tuple[int, int, int]], dict[str, int], dict[int, str], dict[str, tuple[str, str]]]:
    """Build positive playlist-track samples and sampled negatives.

    The negative pool is too large to enumerate. For each playlist, this samples
    track ids uniformly from the catalog until it finds tracks absent from that
    playlist. With the default ratio, the number of negatives equals positives.
    """
    maps = build_track_maps(playlists)
    catalog = np.asarray(list(maps.tid_to_idx.values()), dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    data: list[tuple[int, int, int]] = []
    positive_rows: list[tuple[int, set[int]]] = []

    for playlist_id, playlist in playlists.items():
        playlist_idx = maps.playlist_id_to_idx[playlist_id]
        track_indices = [maps.tid_to_idx[tid] for tid in playlist_tids(playlist) if tid in maps.tid_to_idx]
        seen = set(track_indices)
        for track_idx in track_indices:
            data.append((playlist_idx, int(track_idx), 1))
        positive_rows.append((playlist_idx, seen))

    target_negatives = int(round(len(data) * max(0.0, negative_ratio)))
    if target_negatives == 0:
        return data, maps.tid_to_idx, maps.idx_to_tid, maps.tid_to_meta

    positives_per_playlist = {
        playlist_idx: len(seen)
        for playlist_idx, seen in positive_rows
    }
    total_positives = max(sum(positives_per_playlist.values()), 1)
    added = 0
    for playlist_idx, seen in positive_rows:
        share = positives_per_playlist[playlist_idx] / total_positives
        local_target = int(round(target_negatives * share))
        if added + local_target > target_negatives:
            local_target = target_negatives - added
        for _ in range(max(local_target, 0)):
            while True:
                candidate = int(catalog[int(rng.integers(0, len(catalog)))])
                if candidate not in seen:
                    data.append((playlist_idx, candidate, 0))
                    break
        added += max(local_target, 0)

    while added < target_negatives:
        playlist_idx, seen = positive_rows[int(rng.integers(0, len(positive_rows)))]
        while True:
            candidate = int(catalog[int(rng.integers(0, len(catalog)))])
            if candidate not in seen:
                data.append((playlist_idx, candidate, 0))
                added += 1
                break
    return data, maps.tid_to_idx, maps.idx_to_tid, maps.tid_to_meta


def split_query_targets(
    playlist: list[dict[str, str]],
    query_size: int = 2,
    known_tids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    tids = playlist_tids(playlist)
    if known_tids is not None:
        query = [tid for tid in tids[:query_size] if tid in known_tids]
        targets = [tid for tid in tids[query_size:] if tid in known_tids]
    else:
        query = tids[:query_size]
        targets = tids[query_size:]
    return query, targets


def filter_known_tids(tids: list[str], tid_to_idx: dict[str, int]) -> list[str]:
    return [tid for tid in tids if tid in tid_to_idx]
