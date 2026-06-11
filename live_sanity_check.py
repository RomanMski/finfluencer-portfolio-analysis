from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


REQUIRED_SOURCE_COLUMNS = {
    "video_id",
    "video_title",
    "video_date",
    "ticker",
    "company",
    "action_inferred",
    "event_type",
    "timestamp_seconds",
    "timestamp_url",
    "quote_segment",
    "context_window",
    "confidence_rule_based",
    "source_method",
}


def row_count(path: Path) -> int:
    try:
        return len(pd.read_csv(path))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel",
        default="https://www.youtube.com/@JosephCarlsonShow/videos",
    )
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument(
        "--cookies-from-browser",
        choices=["chrome", "edge", "firefox", "brave", "opera", "vivaldi"],
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="finfluencer_live_check_") as temp_name:
        run_root = Path(temp_name)
        command = [
            sys.executable,
            str(project_root / "run_pipeline.py"),
            "--channel",
            args.channel,
            "--max-videos",
            str(args.max_videos),
            "--root",
            str(run_root),
        ]
        if args.cookies_from_browser:
            command.extend(["--cookies-from-browser", args.cookies_from_browser])
        result = subprocess.run(command, cwd=project_root)
        if result.returncode:
            return result.returncode

        status_path = run_root / "data/transcript_status.csv"
        status = pd.read_csv(status_path)
        downloaded = int(status["transcript_status"].isin(["ok", "already_exists"]).sum())
        candidates_path = run_root / "data/processed/stock_pick_candidates.csv"
        candidate_count = row_count(candidates_path)

        if downloaded == 0:
            reasons = status.get("error_type", pd.Series(dtype=str)).value_counts().to_dict()
            print(f"SKIP: all {len(status)} transcript downloads failed: {reasons}")
            return 0
        if candidate_count == 0:
            print(f"FAIL: {downloaded} transcripts downloaded but candidate extraction returned zero rows.")
            return 1

        candidates = pd.read_csv(candidates_path)
        missing = REQUIRED_SOURCE_COLUMNS.difference(candidates.columns)
        if missing:
            print(f"FAIL: candidate source fields are missing: {sorted(missing)}")
            return 1
        if candidates["timestamp_url"].fillna("").eq("").any():
            print("FAIL: at least one candidate has no timestamp URL.")
            return 1

        print(
            f"PASS: {downloaded} transcripts produced {candidate_count} source-backed candidates "
            f"across {candidates['ticker'].nunique()} tickers."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
