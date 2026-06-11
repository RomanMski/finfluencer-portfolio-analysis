from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_full_research_pipeline import clean_candidates
from run_pipeline import extract_candidates


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "verified_source_audit.json"


class VerifiedSourceAuditTests(unittest.TestCase):
    def test_real_channel_language_does_not_collapse_into_generic_buys(self) -> None:
        cases = [
            (
                "joseph_meta",
                "2026-03-16",
                "I Just Invested $160,000 In This Stock",
                "On January 29th, I bought $45,000 of Meta.",
            ),
            (
                "financial_celh",
                "2026-06-10",
                "How to Get Filthy Rich in the Stock Market",
                "Celsius is one that I'm investing heavy into.",
            ),
            (
                "financial_pltr_history",
                "2026-06-10",
                "How to Get Filthy Rich in the Stock Market",
                "I was buying Palanteer back in 2022 for seven bucks a share.",
            ),
            (
                "financial_sofi",
                "2026-06-10",
                "How to Get Filthy Rich in the Stock Market",
                "I have 0% interest in selling SoFi.",
            ),
            (
                "hkcm_wday",
                "2026-06-10",
                "Diese Aktie ist kaum bekannt",
                "Wir haben aufgenommen, Work Day sind wir eingestiegen.",
            ),
            (
                "hkcm_lulu",
                "2026-06-10",
                "Diese Aktie ist kaum bekannt",
                "Lulu Lemon kommt auf die Shortlist, noch kein Einstieg, Zielzone.",
            ),
            (
                "hkcm_hims",
                "2026-06-10",
                "Diese Aktie ist kaum bekannt",
                "Hims and Hers. Noch kein Einstieg, kurz davor.",
            ),
            (
                "noise_amazon",
                "2025-01-02",
                "Market update",
                "The markets tanked, selling off some companies like Amazon.",
            ),
            (
                "noise_google",
                "2025-01-03",
                "Market update",
                "We will discuss whether or not Google stock is a buy today.",
            ),
            (
                "noise_netflix",
                "2025-01-04",
                "Market history",
                "He bought Netflix after it dropped.",
            ),
            (
                "noise_uber",
                "2025-01-05",
                "Portfolio update",
                "I strongly considered adding Uber but ultimately decided not to.",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_name:
            transcript_dir = Path(temp_name)
            for number, (video_id, date, title, text) in enumerate(cases):
                payload = [
                    {
                        "video_id": video_id,
                        "video_title": title,
                        "video_date": date,
                        "video_url": f"https://www.youtube.com/watch?v={video_id}",
                        "start": float(number * 60),
                        "end": float(number * 60 + 8),
                        "source_method": "verified_youtube_ui",
                        "text": text,
                    }
                ]
                (transcript_dir / f"{video_id}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            candidates = extract_candidates(transcript_dir)

        by_ticker = candidates.set_index("ticker")
        self.assertEqual("personal_trade", by_ticker.loc["META", "evidence_class"])
        self.assertEqual("personal_trade", by_ticker.loc["CELH", "evidence_class"])
        self.assertEqual("personal_holding", by_ticker.loc["SOFI", "evidence_class"])
        self.assertEqual(
            "historical_personal_trade",
            by_ticker.loc["PLTR", "evidence_class"],
        )
        self.assertFalse(bool(by_ticker.loc["PLTR", "strict_eligible"]))
        self.assertEqual("personal_trade", by_ticker.loc["WDAY", "evidence_class"])
        self.assertEqual("planned_or_watchlist", by_ticker.loc["LULU", "evidence_class"])
        self.assertEqual("planned_or_watchlist", by_ticker.loc["HIMS", "evidence_class"])
        self.assertEqual("market_movement", by_ticker.loc["AMZN", "evidence_class"])
        self.assertFalse(bool(by_ticker.loc["GOOGL", "strict_eligible"]))
        self.assertEqual("third_party_action", by_ticker.loc["NFLX", "evidence_class"])
        self.assertEqual("negative_or_rejected", by_ticker.loc["UBER", "evidence_class"])

        strict_events = clean_candidates(candidates)
        self.assertEqual(
            {"META", "CELH", "SOFI", "WDAY"},
            set(strict_events["ticker"]),
        )
        self.assertEqual(
            "2026-01-29",
            strict_events.loc[strict_events["ticker"].eq("META"), "event_date"]
            .iloc[0]
            .date()
            .isoformat(),
        )
        portfolio_tickers = set(
            strict_events.loc[strict_events["use_for_demo_portfolio"], "ticker"]
        )
        self.assertEqual({"META", "CELH", "WDAY"}, portfolio_tickers)

    def test_verified_youtube_fixture_keeps_channel_actions_distinct(self) -> None:
        videos = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_name:
            transcript_dir = Path(temp_name)
            for video in videos:
                segments = []
                for segment in video["segments"]:
                    start = float(segment["start"])
                    segments.append(
                        {
                            "video_id": video["video_id"],
                            "video_title": video["video_title"],
                            "video_date": video["video_date"],
                            "video_url": f"https://www.youtube.com/watch?v={video['video_id']}",
                            "start": start,
                            "end": start + float(segment.get("duration", 0)),
                            "source_method": "verified_youtube_ui",
                            "text": segment["text"],
                        }
                    )
                (transcript_dir / f"{video['video_id']}.json").write_text(
                    json.dumps(segments),
                    encoding="utf-8",
                )
            candidates = extract_candidates(transcript_dir)

        hkcm = candidates[candidates["video_id"].eq("QcdItI0SWEA")]
        strict_hkcm = hkcm[hkcm["strict_eligible"]]
        self.assertEqual({"WDAY"}, set(strict_hkcm["ticker"]))

        meta_dates = set(
            candidates.loc[
                candidates["video_id"].eq("LZ12JplfJoY")
                & candidates["ticker"].eq("META"),
                "event_date",
            ]
        )
        self.assertEqual({"2026-01-29", "2026-02-04", "2026-03-13"}, meta_dates)
        strict_meta_dates = set(
            clean_candidates(candidates)
            .loc[lambda frame: frame["ticker"].eq("META"), "event_date"]
            .dt.date.astype(str)
        )
        self.assertEqual(meta_dates, strict_meta_dates)
        meta_timestamps = set(
            candidates.loc[
                candidates["video_id"].eq("LZ12JplfJoY")
                & candidates["ticker"].eq("META"),
                "timestamp_seconds",
            ]
        )
        self.assertEqual({44, 68, 92}, meta_timestamps)


if __name__ == "__main__":
    unittest.main()
