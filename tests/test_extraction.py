from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_pipeline import CANDIDATE_COLUMNS, extract_candidates
from src.transcripts import classify_transcript_error, parse_ytdlp_json3


class ExtractionTests(unittest.TestCase):
    def test_extracts_source_backed_actions(self) -> None:
        segments = [
            {
                "video_id": "sample123",
                "video_title": "Portfolio update",
                "video_date": "2025-01-02",
                "video_url": "https://www.youtube.com/watch?v=sample123",
                "start": 10.0,
                "end": 14.0,
                "source_method": "test_fixture",
                "text": "I bought more Apple today.",
            },
            {
                "video_id": "sample123",
                "video_title": "Portfolio update",
                "video_date": "2025-01-02",
                "video_url": "https://www.youtube.com/watch?v=sample123",
                "start": 70.0,
                "end": 74.0,
                "source_method": "test_fixture",
                "text": "I sold Tesla and reduced the position.",
            },
            {
                "video_id": "sample123",
                "video_title": "Portfolio update",
                "video_date": "2025-01-02",
                "video_url": "https://www.youtube.com/watch?v=sample123",
                "start": 130.0,
                "end": 134.0,
                "source_method": "test_fixture",
                "text": "Nvidia is on my watchlist for now.",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_name:
            transcript_dir = Path(temp_name)
            (transcript_dir / "sample123.json").write_text(
                json.dumps(segments),
                encoding="utf-8",
            )
            frame = extract_candidates(transcript_dir)

        self.assertEqual(set(CANDIDATE_COLUMNS), set(frame.columns))
        actions = dict(zip(frame["ticker"], frame["action_inferred"]))
        self.assertEqual(actions["AAPL"], "buy_or_add")
        self.assertEqual(actions["TSLA"], "sell_or_reduce")
        self.assertEqual(actions["NVDA"], "watchlist_or_uncertain")
        self.assertTrue(frame["timestamp_url"].str.contains("sample123").all())
        self.assertTrue(frame["quote_segment"].str.len().gt(0).all())
        self.assertTrue(frame["source_method"].eq("test_fixture").all())

    def test_empty_directory_keeps_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            frame = extract_candidates(Path(temp_name))
        self.assertTrue(frame.empty)
        self.assertEqual(CANDIDATE_COLUMNS, frame.columns.tolist())

    def test_parses_ytdlp_json3(self) -> None:
        payload = {
            "events": [
                {
                    "tStartMs": 1250,
                    "dDurationMs": 2200,
                    "segs": [{"utf8": "I bought "}, {"utf8": "Apple"}],
                }
            ]
        }
        segments = parse_ytdlp_json3(payload)
        self.assertEqual(segments[0]["text"], "I bought Apple")
        self.assertEqual(segments[0]["start"], 1.25)
        self.assertEqual(segments[0]["duration"], 2.2)

    def test_classifies_rate_limit(self) -> None:
        self.assertEqual(classify_transcript_error("HTTP Error 429"), "ip_blocked")


if __name__ == "__main__":
    unittest.main()
