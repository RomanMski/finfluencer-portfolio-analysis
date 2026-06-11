from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PREFERRED_LANGUAGES = ["en", "en-US", "en-GB", "de", "de-DE"]
YTDLP_SUBTITLE_LANGUAGES = "en-orig,en,de-orig,de"


def classify_transcript_error(message: str) -> str:
    text = str(message).lower()
    if "ipblocked" in text or "ip blocked" in text or "blocking requests from your ip" in text or "http error 429" in text:
        return "ip_blocked"
    if "transcriptsdisabled" in text or "transcripts are disabled" in text:
        return "transcripts_disabled"
    if "notranscriptfound" in text or "no transcripts were found" in text or "no subtitles" in text:
        return "no_transcript"
    if "video unavailable" in text or "unavailable" in text:
        return "video_unavailable"
    return "download_failed"


def parse_ytdlp_json3(payload: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        pieces = event.get("segs") or []
        text = "".join(str(piece.get("utf8", "")) for piece in pieces)
        text = " ".join(text.replace("\n", " ").split())
        if not text:
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        duration = float(event.get("dDurationMs", 0)) / 1000.0
        segments.append({"text": text, "start": start, "duration": duration})
    return segments


def fetch_with_transcript_api(video_id: str) -> list[dict[str, Any]]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=PREFERRED_LANGUAGES)
    return [
        {
            "text": getattr(item, "text", ""),
            "start": float(getattr(item, "start", 0.0)),
            "duration": float(getattr(item, "duration", 0.0)),
        }
        for item in fetched
    ]


def fetch_with_ytdlp(video_id: str, cookies_from_browser: str | None = None) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="finfluencer_subtitles_") as temp_name:
        temp_dir = Path(temp_name)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            YTDLP_SUBTITLE_LANGUAGES,
            "--sub-format",
            "json3",
            "--no-warnings",
            "-o",
            str(temp_dir / "%(id)s.%(ext)s"),
        ]
        if cookies_from_browser:
            command.extend(["--cookies-from-browser", cookies_from_browser])
        command.append(f"https://www.youtube.com/watch?v={video_id}")

        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        subtitle_files = sorted(temp_dir.glob(f"{video_id}.*.json3"))
        if proc.returncode != 0 or not subtitle_files:
            raise RuntimeError(proc.stdout[-3000:] or "yt-dlp did not produce a subtitle file")

        payload = json.loads(subtitle_files[0].read_text(encoding="utf-8"))
        segments = parse_ytdlp_json3(payload)
        if not segments:
            raise RuntimeError("yt-dlp downloaded an empty subtitle file")
        return segments


def fetch_transcript_segments(
    video_id: str,
    cookies_from_browser: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    errors = []
    try:
        segments = fetch_with_transcript_api(video_id)
        if segments:
            return segments, "youtube_transcript_api"
    except Exception as exc:
        errors.append(f"youtube-transcript-api: {exc}")

    try:
        segments = fetch_with_ytdlp(video_id, cookies_from_browser=cookies_from_browser)
        if segments:
            return segments, "yt_dlp_subtitles"
    except Exception as exc:
        errors.append(f"yt-dlp: {exc}")

    raise RuntimeError("\n\n".join(errors) or "No transcript source returned data")
