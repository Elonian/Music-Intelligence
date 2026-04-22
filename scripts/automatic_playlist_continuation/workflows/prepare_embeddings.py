#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io_helpers import save_json  # noqa: E402
from utils.playlist_continuation import collect_unique_tids, ensure_embeddings_extracted, load_playlists  # noqa: E402
from utils.project_paths import (  # noqa: E402
    AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_DIR,
    AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_ZIP,
    AUTOMATIC_PLAYLIST_CONTINUATION_METRICS_DIR,
    AUTOMATIC_PLAYLIST_CONTINUATION_TEST_JSON,
    AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and summarize automatic playlist continuation audio embeddings.")
    parser.add_argument("--zip-path", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_ZIP)
    parser.add_argument("--embedding-dir", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_EMBEDDING_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--playlist-tracks-only", action="store_true")
    parser.add_argument("--train-json", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON)
    parser.add_argument("--test-json", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_TEST_JSON)
    parser.add_argument("--summary-path", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_METRICS_DIR / "embedding_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_tids = None
    if args.playlist_tracks_only:
        train_playlists = load_playlists(args.train_json)
        test_playlists = load_playlists(args.test_json)
        requested_tids = sorted(set(collect_unique_tids(train_playlists)) | set(collect_unique_tids(test_playlists)))
    embedding_dir = ensure_embeddings_extracted(
        args.embedding_dir,
        args.zip_path,
        overwrite=args.overwrite,
        track_ids=requested_tids,
    )
    with ZipFile(args.zip_path) as zip_file:
        real_files = [
            member
            for member in zip_file.infolist()
            if member.filename.startswith("audio_embeddings/")
            and member.filename.endswith(".npy")
            and not member.is_dir()
        ]
        selected_names = None if requested_tids is None else {f"{tid}.npy" for tid in requested_tids}
        selected_files = [
            member for member in real_files if selected_names is None or Path(member.filename).name in selected_names
        ]
        extracted_names = {path.name for path in embedding_dir.glob("*.npy") if path.stat().st_size > 0}
        summary = {
            "zip_path": str(args.zip_path),
            "embedding_dir": str(embedding_dir),
            "real_embedding_files": len(real_files),
            "selected_embedding_files": len(selected_files),
            "requested_track_ids": None if requested_tids is None else len(requested_tids),
            "compressed_bytes": int(sum(member.compress_size for member in real_files)),
            "uncompressed_bytes": int(sum(member.file_size for member in real_files)),
            "extracted_files": len(extracted_names),
            "selected_files_present": sum(1 for member in selected_files if Path(member.filename).name in extracted_names),
        }
    save_json(args.summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
