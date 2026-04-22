from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from zipfile import ZipFile

import numpy as np

from utils.io_helpers import ensure_dir
from utils.project_paths import (
    AUTOMATIC_PLAYLIST_CONTINUATION_DATA_ROOT,
    AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_DIR,
    AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_ZIP,
    AUTOMATIC_PLAYLIST_CONTINUATION_TEST_JSON,
    AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON,
)


PlaylistDict = dict[str, list[dict[str, str]]]


def load_playlists(path: Path | str) -> PlaylistDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a playlist dictionary in {path}.")
    return payload


def load_train_playlists(data_root: Path | str | None = None) -> PlaylistDict:
    root = Path(data_root) if data_root is not None else AUTOMATIC_PLAYLIST_CONTINUATION_DATA_ROOT
    return load_playlists(root / "train_playlists.json" if data_root is not None else AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON)


def load_test_playlists(data_root: Path | str | None = None) -> PlaylistDict:
    root = Path(data_root) if data_root is not None else AUTOMATIC_PLAYLIST_CONTINUATION_DATA_ROOT
    return load_playlists(root / "test_playlists.json" if data_root is not None else AUTOMATIC_PLAYLIST_CONTINUATION_TEST_JSON)


def playlist_tids(playlist: list[dict[str, str]] | list[str]) -> list[str]:
    if not playlist:
        return []
    first = playlist[0]
    if isinstance(first, str):
        return [str(item) for item in playlist]
    return [str(item["tid"]) for item in playlist if "tid" in item]


def collect_playlist_metadata(playlists: PlaylistDict) -> dict[str, tuple[str, str]]:
    metadata: dict[str, tuple[str, str]] = {}
    for playlist in playlists.values():
        for track in playlist:
            tid = str(track["tid"])
            metadata[tid] = (str(track.get("artist_name", "")), str(track.get("track_name", "")))
    return metadata


def collect_unique_tids(playlists: PlaylistDict) -> list[str]:
    tids = set()
    for playlist in playlists.values():
        tids.update(playlist_tids(playlist))
    return sorted(tids)


def resolve_embedding_dir(path: Path | str | None = None) -> Path:
    if path is None:
        return AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_DIR
    candidate = Path(path).expanduser().resolve()
    if candidate.name == "audio_embeddings":
        return candidate
    if (candidate / "audio_embeddings").exists():
        return candidate / "audio_embeddings"
    return candidate


def ensure_embeddings_extracted(
    embedding_dir: Path | str | None = None,
    zip_path: Path | str | None = None,
    overwrite: bool = False,
    track_ids: Iterable[str] | None = None,
) -> Path:
    destination = resolve_embedding_dir(embedding_dir)
    archive = Path(zip_path) if zip_path is not None else AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_ZIP
    requested_names = None if track_ids is None else {f"{tid}.npy" for tid in track_ids}
    if not archive.exists():
        raise FileNotFoundError(f"Missing embedding archive: {archive}")
    with ZipFile(archive) as zip_file:
        members = [
            member
            for member in zip_file.infolist()
            if member.filename.startswith("audio_embeddings/")
            and member.filename.endswith(".npy")
            and not member.is_dir()
            and (requested_names is None or Path(member.filename).name in requested_names)
        ]
        if destination.exists() and not overwrite:
            existing_names = {path.name for path in destination.glob("*.npy") if path.stat().st_size > 0}
            required_names = {Path(member.filename).name for member in members}
            if required_names.issubset(existing_names):
                return destination
        ensure_dir(destination)
        for member in members:
            target = destination / Path(member.filename).name
            if target.exists() and target.stat().st_size > 0 and not overwrite:
                continue
            with zip_file.open(member) as source, target.open("wb") as output:
                output.write(source.read())
    required_names = {Path(member.filename).name for member in members}
    extracted_names = {path.name for path in destination.glob("*.npy") if path.stat().st_size > 0}
    missing_names = required_names - extracted_names
    if missing_names:
        raise RuntimeError(
            f"Expected {len(required_names)} embedding files in {destination}, missing {len(missing_names)}."
        )
    return destination


def load_track_embedding(tid: str, embedding_dir: Path | str, allow_missing: bool = False) -> np.ndarray | None:
    path = resolve_embedding_dir(embedding_dir) / f"{tid}.npy"
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    try:
        embedding = np.load(path)
    except (EOFError, OSError, ValueError):
        if allow_missing:
            return None
        raise
    if embedding.ndim == 1:
        return embedding.astype(np.float32, copy=False)
    if embedding.ndim != 2:
        raise ValueError(f"Expected 1D or 2D embedding for {tid}, got shape {embedding.shape}.")
    return embedding.astype(np.float32, copy=False).mean(axis=0)


def summarize_playlist_collection(playlists: PlaylistDict) -> dict:
    lengths = [len(playlist) for playlist in playlists.values()]
    tids = collect_unique_tids(playlists)
    if not lengths:
        return {"playlists": 0, "track_rows": 0, "unique_tracks": 0}
    return {
        "playlists": len(playlists),
        "track_rows": int(sum(lengths)),
        "unique_tracks": len(tids),
        "min_length": int(min(lengths)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "max_length": int(max(lengths)),
    }
