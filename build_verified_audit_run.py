from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_pipeline import create_charts, extract_candidates, make_review_template


PROJECT_ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "verified_source_audit.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holding-days", type=int, default=63)
    parser.add_argument("--refresh-prices", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = PROJECT_ROOT / "runs" / f"{stamp}_verified-source-audit"
    transcript_dir = run_dir / "data" / "transcripts"
    processed_dir = run_dir / "data" / "processed"
    for path in [
        transcript_dir,
        processed_dir,
        run_dir / "data" / "market",
        run_dir / "outputs" / "charts",
        run_dir / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    source_videos = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    inventory_rows = []
    status_rows = []

    for video in source_videos:
        video_id = video["video_id"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        enriched = []
        for segment in video["segments"]:
            start = float(segment["start"])
            duration = float(segment.get("duration", 0))
            enriched.append(
                {
                    "video_id": video_id,
                    "video_title": video["video_title"],
                    "video_date": video["video_date"],
                    "video_url": video_url,
                    "start": start,
                    "end": start + duration,
                    "timestamp": f"{int(start // 60):02d}:{int(start % 60):02d}",
                    "timestamp_url": f"{video_url}&t={int(start)}s",
                    "source_method": "verified_youtube_ui",
                    "text": segment["text"],
                }
            )

        (transcript_dir / f"{video_id}.json").write_text(
            json.dumps(enriched, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        inventory_rows.append(
            {
                "video_id": video_id,
                "date": video["video_date"],
                "title": video["video_title"],
                "url": video_url,
                "channel": video["channel"],
            }
        )
        status_rows.append(
            {
                "video_id": video_id,
                "transcript_status": "ok",
                "segments": len(enriched),
                "source_method": "verified_youtube_ui",
                "error_type": "",
                "error": "",
            }
        )

    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(run_dir / "data" / "video_inventory.csv", index=False)
    pd.DataFrame(status_rows).to_csv(run_dir / "data" / "transcript_status.csv", index=False)

    candidates = extract_candidates(transcript_dir)
    candidates.to_csv(processed_dir / "stock_pick_candidates.csv", index=False)
    make_review_template(candidates).to_csv(
        processed_dir / "stock_pick_candidates_review_template.csv",
        index=False,
    )
    create_charts(candidates, run_dir / "outputs" / "charts")

    manifest = {
        "channel": "Verified source audit: Joseph Carlson + Financial Education + HKCM",
        "max_videos": len(source_videos),
        "holding_days": args.holding_days,
        "requested_mode": "Strict portfolio mode",
        "mode": "running",
        "ok": False,
        "message": "Building verified source audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_full_research_pipeline.py"),
        "--channel",
        manifest["channel"],
        "--max-videos",
        str(len(source_videos)),
        "--holding-days",
        str(args.holding_days),
        "--root",
        str(run_dir),
    ]
    if args.refresh_prices:
        command.append("--refresh-prices")
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    events_path = processed_dir / "clean_candidate_events.csv"
    daily_path = processed_dir / "event_portfolio_daily.csv"
    events = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
    daily = pd.read_csv(daily_path) if daily_path.exists() else pd.DataFrame()
    manifest.update(
        {
            "mode": "strict" if not daily.empty else "diagnostic",
            "ok": not daily.empty,
            "message": "Verified source audit completed",
            "stats": {
                "videos": len(inventory),
                "transcripts": len(status_rows),
                "transcript_failures": 0,
                "candidates": len(candidates),
                "events": len(events),
                "portfolio_days": len(daily),
            },
        }
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
