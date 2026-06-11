from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from loose_research_pipeline import build_events
from run_full_research_pipeline import (
    build_event_portfolio,
    compute_forward_returns,
    compute_metrics,
)


class ResearchPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.bdate_range("2025-01-02", periods=140)
        self.prices = pd.DataFrame(
            {
                "AAPL": 100 * np.cumprod(np.full(len(index), 1.001)),
                "MSFT": 200 * np.cumprod(np.full(len(index), 1.0005)),
                "SPY": 400 * np.cumprod(np.full(len(index), 1.0004)),
                "QQQ": 350 * np.cumprod(np.full(len(index), 1.0006)),
            },
            index=index,
        )
        self.events = pd.DataFrame(
            [
                {
                    "video_date": pd.Timestamp("2025-01-03"),
                    "ticker": "AAPL",
                    "event_type": "buy_or_add",
                    "use_for_demo_portfolio": True,
                },
                {
                    "video_date": pd.Timestamp("2025-02-03"),
                    "ticker": "MSFT",
                    "event_type": "buy_or_add",
                    "use_for_demo_portfolio": True,
                },
            ]
        )

    def test_builds_portfolio_metrics_and_forward_returns(self) -> None:
        daily = build_event_portfolio(self.events, self.prices, holding_days=63)
        metrics = compute_metrics(daily)
        forward = compute_forward_returns(self.events, self.prices)

        self.assertFalse(daily.empty)
        self.assertIn("event_portfolio_growth", daily.columns)
        self.assertEqual({"Event portfolio", "SPY", "QQQ"}, set(metrics["series"]))
        self.assertEqual(2, len(forward))
        self.assertIn("return_1m", forward.columns)

    def test_strict_portfolio_requires_explicit_buy(self) -> None:
        events = self.events.copy()
        events["use_for_demo_portfolio"] = False
        daily = build_event_portfolio(events, self.prices, holding_days=63)
        self.assertTrue(daily.empty)

    def test_loose_mode_excludes_negative_buy_language_and_fills_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "video_id": "one",
                        "video_date": "",
                        "ticker": "AAPL",
                        "action_inferred": "negative_or_not_buying",
                        "quote_segment": "I would not buy Apple here.",
                        "confidence_rule_based": 0.9,
                    },
                    {
                        "video_id": "two",
                        "video_date": "",
                        "ticker": "MSFT",
                        "action_inferred": "buy_or_add",
                        "quote_segment": "I bought Microsoft.",
                        "confidence_rule_based": 0.9,
                    },
                ]
            ).to_csv(processed / "stock_pick_candidates.csv", index=False)
            pd.DataFrame(
                [
                    {"video_id": "one", "date": "2025-01-02"},
                    {"video_id": "two", "date": "2025-01-03"},
                ]
            ).to_csv(root / "data" / "video_inventory.csv", index=False)

            events = build_events(root)

        self.assertEqual(["MSFT"], events["ticker"].tolist())
        self.assertEqual(pd.Timestamp("2025-01-03"), events.iloc[0]["video_date"])


if __name__ == "__main__":
    unittest.main()
