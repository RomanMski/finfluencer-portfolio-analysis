from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from src.ticker_universe import TICKERS
from src.transcripts import classify_transcript_error, fetch_transcript_segments


CANDIDATE_COLUMNS = [
    "video_id",
    "video_date",
    "event_date",
    "event_date_text",
    "date_source",
    "video_title",
    "video_url",
    "timestamp_seconds",
    "timestamp_start_seconds",
    "timestamp_end_seconds",
    "timestamp_start",
    "timestamp_end",
    "timestamp_url",
    "ticker",
    "company",
    "action_inferred",
    "event_type",
    "evidence_class",
    "strict_eligible",
    "confidence_rule_based",
    "source_method",
    "quote_segment",
    "asset_context",
    "context_window",
    "manual_review",
    "verified_action",
    "include_in_portfolio",
    "notes",
]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{proc.stderr[:4000]}"
        )
    return proc.stdout


def get_video_inventory(channel_url: str, max_videos: int, out_csv: Path) -> pd.DataFrame:
    """
    Uses yt-dlp in flat-playlist mode to collect video metadata without downloading videos.
    """
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--dump-json",
        "--playlist-end",
        str(max_videos),
        channel_url,
    ]
    raw = run_cmd(cmd)
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = item.get("id")
        if not video_id:
            continue

        upload_date = item.get("upload_date")
        if upload_date and re.fullmatch(r"\d{8}", str(upload_date)):
            date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        else:
            date = item.get("timestamp")
            if date:
                date = datetime.utcfromtimestamp(date).strftime("%Y-%m-%d")
            else:
                date = ""

        rows.append(
            {
                "video_id": video_id,
                "date": date,
                "title": item.get("title", ""),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "duration": item.get("duration", ""),
                "view_count": item.get("view_count", ""),
                "channel": item.get("channel", ""),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["video_id"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def seconds_to_hhmmss(seconds: float | int) -> str:
    seconds = int(max(0, float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def save_transcripts(
    inventory: pd.DataFrame,
    transcript_dir: Path,
    cookies_from_browser: str | None = None,
    request_delay: float = 0.75,
) -> pd.DataFrame:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    started = time.monotonic()
    total = len(inventory)
    consecutive_ip_blocks = 0

    for number, (_, row) in enumerate(inventory.iterrows(), start=1):
        video_id = row["video_id"]
        json_path = transcript_dir / f"{video_id}.json"
        txt_path = transcript_dir / f"{video_id}.txt"

        if json_path.exists() and txt_path.exists():
            status_rows.append(
                {
                    "video_id": video_id,
                    "transcript_status": "already_exists",
                    "segments": "",
                    "source_method": "saved_file",
                    "error_type": "",
                    "error": "",
                }
            )
            consecutive_ip_blocks = 0
        elif consecutive_ip_blocks >= 3 and not cookies_from_browser:
            status_rows.append(
                {
                    "video_id": video_id,
                    "transcript_status": "failed",
                    "segments": 0,
                    "source_method": "",
                    "error_type": "ip_blocked",
                    "error": "Skipped after three consecutive YouTube IP blocks in this run.",
                }
            )
        else:
            try:
                segments, source_method = fetch_transcript_segments(
                    video_id,
                    cookies_from_browser=cookies_from_browser,
                )
                enriched = []
                for seg in segments:
                    start = float(seg.get("start", 0.0))
                    duration = float(seg.get("duration", 0.0))
                    enriched.append(
                        {
                            "video_id": video_id,
                            "video_title": row.get("title", ""),
                            "video_date": row.get("date", ""),
                            "video_url": row.get("url", ""),
                            "start": start,
                            "end": start + duration,
                            "timestamp": seconds_to_hhmmss(start),
                            "timestamp_url": f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s",
                            "source_method": source_method,
                            "text": clean_text(seg.get("text", "")),
                        }
                    )

                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(enriched, f, ensure_ascii=False, indent=2)

                with txt_path.open("w", encoding="utf-8") as f:
                    f.write(f"TITLE: {row.get('title', '')}\n")
                    f.write(f"DATE: {row.get('date', '')}\n")
                    f.write(f"URL: {row.get('url', '')}\n\n")
                    for seg in enriched:
                        f.write(f"[{seg['timestamp']}] {seg['text']}\n")

                status_rows.append(
                    {
                        "video_id": video_id,
                        "transcript_status": "ok",
                        "segments": len(enriched),
                        "source_method": source_method,
                        "error_type": "",
                        "error": "",
                    }
                )
                consecutive_ip_blocks = 0
            except Exception as exc:
                error = str(exc)
                error_type = classify_transcript_error(error)
                status_rows.append(
                    {
                        "video_id": video_id,
                        "transcript_status": "failed",
                        "segments": 0,
                        "source_method": "",
                        "error_type": error_type,
                        "error": error[:1000],
                    }
                )
                consecutive_ip_blocks = consecutive_ip_blocks + 1 if error_type == "ip_blocked" else 0

        elapsed = max(time.monotonic() - started, 0.001)
        rate = number / elapsed
        eta = int((total - number) / rate) if rate else 0
        print(
            f"PROGRESS transcripts {number} {total} elapsed={int(elapsed)} eta={eta}",
            flush=True,
        )
        if number < total and request_delay > 0 and consecutive_ip_blocks < 3:
            time.sleep(request_delay)

    return pd.DataFrame(status_rows)


def zip_transcripts(transcript_dir: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(transcript_dir.glob("*")):
            if path.suffix.lower() in {".txt", ".json"}:
                z.write(path, arcname=path.name)


def compile_patterns() -> list[tuple[str, str, list[re.Pattern]]]:
    """
    Returns (ticker, company, patterns).
    Patterns include ticker and company aliases, but avoid too many ambiguous tickers.
    """
    compiled = []
    for ticker, company, aliases in TICKERS:
        pats = []
        # Ticker match: requires common stock-style context or cashtag to reduce false positives.
        # We still allow bare uppercase tickers for low-ambiguity symbols.
        if len(ticker) >= 3:
            pats.append(re.compile(rf"(?<![A-Za-z0-9])\$?{re.escape(ticker)}(?![A-Za-z0-9])"))
        else:
            pats.append(re.compile(rf"(?<![A-Za-z0-9])\${re.escape(ticker)}(?![A-Za-z0-9])"))
        for alias in aliases + [company]:
            if alias:
                pats.append(re.compile(rf"\b{re.escape(alias)}\b", flags=re.IGNORECASE))
        compiled.append((ticker, company, pats))
    return compiled


BUY_PATTERNS = [
    r"\b(buy|buying|bought|purchase|purchasing|add|adding|added|accumulate|building.*position|start.*position|opened.*position|initiate|initiated)\b",
    r"\b(stocks? to buy|best stocks?|top stocks?|undervalued|cheap|attractive valuation|strong buy)\b",
]
SELL_PATTERNS = [
    r"\b(sell|selling|sold|trim|trimming|trimmed|reduce|reducing|reduced|exit|exited|close.*position|closed.*position)\b",
]
HOLD_PATTERNS = [
    r"\b(hold|holding|holdings|own|owned|still own|portfolio|position|largest position|allocation|weight|dividend income)\b",
]
WATCH_PATTERNS = [
    r"\b(watchlist|watching|looking at|considering|maybe|could buy|might buy|interested in)\b",
]
NEGATION_PATTERNS = [
    r"\b(not buying|wouldn't buy|would not buy|don't buy|do not buy|avoid|too expensive|overvalued|not interested)\b",
]


BUY_PATTERNS.extend([
    r"\b(long|long setup|long entry|entry zone|accumulation zone|support buy|breakout buy|buy zone|adding zone)\b",
    r"\b(kaufen|kaufe|kauft|gekauft|nachkaufen|aufstocken|einstieg|einsteigen|eingestiegen|kaufzone|einstiegsbereich|akkumulieren)\b",
    r"\b(achat|acheter|acheté|acheterais|zone d'achat|entrée|position longue)\b",
])
SELL_PATTERNS.extend([
    r"\b(short|short setup|short entry|sell zone|take profit|profit taking|stop loss|stop-loss|exit zone|resistance sell)\b",
    r"\b(verkaufen|verkaufe|verkauft|reduzieren|abbauen|ausstieg|aussteigen|ausgestiegen|gewinnmitnahme|gewinne mitnehmen)\b",
    r"\b(vendre|vendu|vente|sortie|réduire|prise de profit)\b",
])
HOLD_PATTERNS.extend([
    r"\b(position|allocation|portfolio update|holding update|current holding|still holding|not selling)\b",
    r"\b(halten|halte|hält|haelt|gehalten|positionierung|bestand|weiter halten|nicht verkaufen)\b",
    r"\b(garder|conserver|position actuelle|portefeuille)\b",
    r"\b(target zone|price target|support|resistance|trend line|chart setup|technical setup|elliott wave|wave count)\b",
    r"\b(zielzone|kursziel|unterstützung|unterstuetzung|widerstand|trendlinie|chartanalyse|elliott wave|welle)\b",
])
WATCH_PATTERNS.extend([
    r"\b(watchlist|watching|scenario|if.*then|possible setup|monitoring|waiting for confirmation)\b",
    r"\b(beobachten|abwarten|szenario|möglich|moeglich|könnte|koennte|wenn.*dann|setup beobachten)\b",
    r"\b(surveiller|scénario|possible|attendre|confirmation)\b",
])
NEGATION_PATTERNS.extend([
    r"\b(not buying|no buy|would not buy|avoid|too risky|overvalued|bad setup)\b",
    r"\b(nicht kaufen|kein kauf|würde nicht kaufen|wuerde nicht kaufen|finger weg|zu riskant|überbewertet|ueberbewertet)\b",
    r"\b(ne pas acheter|éviter|trop risqué|surévalué)\b",
])

PERSONAL_BUY_PATTERNS = [
    r"\b(?:i|we)\s+(?:just\s+|recently\s+|already\s+|have\s+|had\s+|originally\s+|continue(?:d)?\s+to\s+)*(?:bought|purchased|added|accumulated|invested|entered|initiated|opened)\b",
    r"\b(?:i'm|i am|we're|we are)\s+(?:still\s+|currently\s+)?investing\b",
    r"\b(?:i|we)\s+(?:was|were)\s+(?:buying|adding|investing)\b",
    r"\b(?:my|our)\s+(?:latest\s+|recent\s+|new\s+)?(?:buy|purchase|entry)\b",
    r"\banother\s+\$[\d,.]+\s+(?:worth\s+)?of\b",
    r"\bwir\s+sind\s+(?:bereits\s+)?eingestiegen\b",
    r"\bsind\s+wir\s+(?:bereits\s+)?eingestiegen\b",
    r"\b(?:ich|wir)\s+hab(?:e|en)\s+(?:bereits\s+|neu\s+)?(?:gekauft|nachgekauft|aufgestockt)\b",
    r"\b(?:j'ai|nous avons)\s+(?:achete|acheté|ajoute|ajouté|investi)\b",
]
PERSONAL_SELL_PATTERNS = [
    r"\b(?:i|we)\s+(?:just\s+|recently\s+|have\s+|had\s+)*(?:sold|trimmed|reduced|exited|closed)\b",
    r"\b(?:i'm|i am|we're|we are)\s+(?:selling|trimming|reducing|exiting)\b",
    r"\b(?:my|our)\s+(?:latest\s+|recent\s+)?(?:sale|trim|exit)\b",
    r"\b(?:ich|wir)\s+hab(?:e|en)\s+(?:verkauft|reduziert|abgebaut)\b",
]
PERSONAL_HOLD_PATTERNS = [
    r"\b(?:i|we)\s+(?:still\s+|currently\s+)?(?:own|hold)\b",
    r"\b(?:i|we)\s+have\b.{0,50}\b(?:shares?|position|holding|stock)\b",
    r"\b(?:i'm|i am|we're|we are)\s+not\s+selling\b",
    r"\b(?:my|our)\s+(?:current\s+)?(?:portfolio|position|holding|shares|allocation)\b",
    r"\b(?:i|we)\s+have\s+\d[\d,.]*\s+shares\b",
    r"\b(?:i|we)\s+(?:have|had)\s+(?:zero|no|0%)\s+interest\s+in\s+selling\b",
    r"\b(?:ich|wir)\s+(?:halte|halten|besitze|besitzen)\b",
    r"\b(?:mein|unser)(?:e|er|en)?\s+(?:depot|portfolio|position|bestand)\b",
]
EXPLICIT_RECOMMENDATION_PATTERNS = [
    r"\b(?:strong|easy|clear|obvious)\s+buy\b",
    r"\b(?:i|we)\s+(?:would|will)\s+buy\b",
    r"\b(?:i|we)\s+(?:think|believe|consider)\b.{0,80}\b(?:is|looks like)\s+(?:a\s+)?buy\b",
    r"\b(?:you|investors?)\s+should\s+buy\b",
    r"\b(?:recommend|recommended|recommendation)\b.{0,40}\b(?:buy|buying)\b",
    r"\b(?:klare|starke|eindeutige)\s+kauf(?:chance|empfehlung)?\b",
]
PLANNED_OR_WATCH_PATTERNS = [
    r"\b(?:watchlist|watching|considering|considered|planning|plan to|might buy|could buy|possible entry|entry zone|target zone|waiting for confirmation)\b",
    r"\b(?:will|would)\s+(?:add|enter|start|open)\b",
    r"\b(?:not yet|no)\s+(?:buy|entry|position)\b",
    r"\b(?:noch kein einstieg|demn[aä]chst aufnehmen|werden .*einsteigen|werden .*aufnehmen|zielzone|einstiegszone|shortlist|potenziell)\b",
]
NOT_BUY_PATTERNS = [
    r"\b(?:not|never)\s+buy(?:ing)?\b",
    r"\b(?:wouldn't|would not|don't|do not|didn't|did not|can't|cannot)\b.{0,50}\b(?:buy(?:ing)?|purchas(?:e|ing)|add(?:ing)?|invest(?:ing)?)\b",
    r"\b(?:decided|chose)\s+not\s+to\b.{0,40}\b(?:buy|add|invest)\b",
    r"\bultimately\s+decided\s+not\s+to\b",
    r"\btoo\s+(?:expensive|risky)\s+to\s+buy\b",
    r"\b(?:nicht kaufen|kein kauf|finger weg|zu riskant|ueberbewertet|überbewertet)\b",
]
MARKET_MOVEMENT_PATTERNS = [
    r"\b(?:stock|shares?|market|company|price)\s+(?:has\s+|have\s+|is\s+|are\s+|was\s+|were\s+)?(?:sold off|selling off|traded down|down|dropping|falling)\b",
    r"\b(?:sell-off|selloff|selling off|sold down|getting traded down)\b",
]
THIRD_PARTY_ACTION_PATTERNS = [
    r"\b(?:he|she|they|investors?|workers?|analysts?|buffett|burry|ackman|wood)\b.{0,100}\b(?:bought|buying|sold|selling|added|trimmed|reduced|invested)\b",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_action_evidence(context: str) -> tuple[str, float, str, bool]:
    """Classify creator evidence without treating market chatter as a creator trade."""
    ctx = clean_text(context).lower().replace("’", "'")

    if _matches_any(NOT_BUY_PATTERNS, ctx):
        return "negative_or_not_buying", 0.92, "negative_or_rejected", False
    if _matches_any(PERSONAL_SELL_PATTERNS, ctx):
        return "sell_or_reduce", 0.96, "personal_trade", True
    if _matches_any(PERSONAL_BUY_PATTERNS, ctx):
        return "buy_or_add", 0.96, "personal_trade", True
    if _matches_any(PLANNED_OR_WATCH_PATTERNS, ctx):
        return "watchlist_or_uncertain", 0.86, "planned_or_watchlist", False
    if _matches_any(MARKET_MOVEMENT_PATTERNS, ctx):
        return "mention_only", 0.88, "market_movement", False
    if _matches_any(THIRD_PARTY_ACTION_PATTERNS, ctx):
        return "mention_only", 0.88, "third_party_action", False
    if _matches_any(EXPLICIT_RECOMMENDATION_PATTERNS, ctx):
        return "buy_or_add", 0.90, "explicit_recommendation", True
    if _matches_any(PERSONAL_HOLD_PATTERNS, ctx):
        return "hold_or_portfolio_holding", 0.90, "personal_holding", True

    if _matches_any(SELL_PATTERNS, ctx):
        return "sell_or_reduce", 0.45, "generic_action_language", False
    if _matches_any(BUY_PATTERNS, ctx):
        return "buy_or_add", 0.45, "generic_action_language", False
    if _matches_any(HOLD_PATTERNS, ctx):
        return "hold_or_portfolio_holding", 0.40, "generic_holding_language", False
    if _matches_any(WATCH_PATTERNS, ctx):
        return "watchlist_or_uncertain", 0.40, "planned_or_watchlist", False
    return "mention_only", 0.25, "mention_only", False


def infer_action(context: str) -> tuple[str, float]:
    action, confidence, _, _ = classify_action_evidence(context)
    return action, confidence


def asset_local_context(context: str, patterns: list[re.Pattern], radius: int = 240) -> str:
    spans = [match.span() for pattern in patterns for match in pattern.finditer(context)]
    if not spans:
        return clean_text(context)

    clauses = []
    for match_start, match_end in spans:
        boundaries_before = [
            context.rfind(mark, 0, match_start)
            for mark in [".", "!", "?", ";"]
        ]
        clause_start = max(boundaries_before) + 1
        boundaries_after = [
            position
            for mark in [".", "!", "?", ";"]
            if (position := context.find(mark, match_end)) >= 0
        ]
        clause_end = min(boundaries_after) + 1 if boundaries_after else len(context)
        clause = clean_text(context[clause_start:clause_end])
        if clause and clause not in clauses:
            clauses.append(clause)

    selected = clean_text(" ".join(clauses))
    if selected:
        return selected

    start = max(0, min(span[0] for span in spans) - radius)
    end = min(len(context), max(span[1] for span in spans) + radius)
    return clean_text(context[start:end])


def action_adjacent_context(context: str, local_context: str) -> str:
    start = context.find(local_context)
    if start < 0:
        return local_context
    end = start + len(local_context)
    after = context[end:].lstrip(" .!?;")
    next_clause = re.split(r"[.!?;]", after, maxsplit=1)[0].strip()
    if re.match(
        r"^(?:i|we|my|our|ich|wir|mein|unser|not|no|noch|when|sobald|this|it|bought|sold|entered|eingestiegen|gekauft|verkauft)\b",
        next_clause,
        flags=re.IGNORECASE,
    ):
        return clean_text(f"{local_context} {next_clause}")
    return local_context


MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def infer_event_date(context: str, video_date: str) -> tuple[str, str, str]:
    try:
        upload_date = pd.Timestamp(video_date).normalize()
    except Exception:
        upload_date = pd.NaT

    month_pattern = "|".join(MONTH_NUMBERS)
    match = re.search(
        rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b",
        context,
        flags=re.IGNORECASE,
    )
    if match:
        year = int(match.group(3)) if match.group(3) else (
            int(upload_date.year) if pd.notna(upload_date) else datetime.now().year
        )
        month = MONTH_NUMBERS[match.group(1).lower()]
        if pd.notna(upload_date) and not match.group(3) and month > upload_date.month + 1:
            year -= 1
        try:
            event_date = pd.Timestamp(year=year, month=month, day=int(match.group(2)))
            return event_date.date().isoformat(), match.group(0), "spoken_exact_date"
        except ValueError:
            pass

    iso_match = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", context)
    if iso_match:
        try:
            event_date = pd.Timestamp(iso_match.group(0))
            return event_date.date().isoformat(), iso_match.group(0), "spoken_exact_date"
        except ValueError:
            pass

    if pd.notna(upload_date) and re.search(r"\byesterday\b", context, flags=re.IGNORECASE):
        return (upload_date - pd.Timedelta(days=1)).date().isoformat(), "yesterday", "spoken_relative_date"
    if pd.notna(upload_date):
        return upload_date.date().isoformat(), "", "video_upload_date"
    return "", "", "missing_date"


def build_context(
    segments: list[dict[str, Any]],
    idx: int,
    window: int = 2,
    max_gap_seconds: float = 20.0,
) -> tuple[str, float, float]:
    lo = idx
    for candidate in range(idx - 1, max(-1, idx - window - 1), -1):
        candidate_end = float(segments[candidate].get("end", segments[candidate].get("start", 0.0)))
        next_start = float(segments[candidate + 1].get("start", 0.0))
        if next_start - candidate_end > max_gap_seconds:
            break
        lo = candidate

    hi = idx + 1
    for candidate in range(idx + 1, min(len(segments), idx + window + 1)):
        previous_end = float(segments[candidate - 1].get("end", segments[candidate - 1].get("start", 0.0)))
        candidate_start = float(segments[candidate].get("start", 0.0))
        if candidate_start - previous_end > max_gap_seconds:
            break
        hi = candidate + 1

    context = " ".join(clean_text(s.get("text", "")) for s in segments[lo:hi])
    start = float(segments[lo].get("start", 0.0))
    end = float(segments[hi - 1].get("end", segments[hi - 1].get("start", 0.0)))
    return clean_text(context), start, end


def extract_candidates(transcript_dir: Path) -> pd.DataFrame:
    patterns = compile_patterns()
    rows = []

    for json_path in sorted(transcript_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            segments = json.load(f)

        if not segments:
            continue

        for i, seg in enumerate(segments):
            text = clean_text(seg.get("text", ""))
            if not text:
                continue

            matched = []
            for ticker, company, pats in patterns:
                if any(p.search(text) for p in pats):
                    matched.append((ticker, company, pats))

            if not matched:
                continue

            context, start, end = build_context(segments, i, window=2)
            video_id = seg.get("video_id", "")
            quote_start = float(seg.get("start", start))
            quote_end = float(seg.get("end", quote_start))

            for ticker, company, pats in matched:
                local_quote = asset_local_context(text, pats, radius=160)
                action, conf, evidence_class, strict_eligible = classify_action_evidence(local_quote)
                local_context = local_quote
                if evidence_class == "mention_only":
                    adjacent_context = action_adjacent_context(context, local_quote)
                    adjacent_result = classify_action_evidence(adjacent_context)
                    if adjacent_result[2] != "mention_only":
                        action, conf, evidence_class, strict_eligible = adjacent_result
                        local_context = adjacent_context
                event_date, event_date_text, date_source = infer_event_date(
                    local_context,
                    seg.get("video_date", ""),
                )
                if date_source == "video_upload_date" and re.search(
                    r"\b(?:originally|that purchase|this purchase)\b",
                    local_context,
                    flags=re.IGNORECASE,
                ):
                    context_date = infer_event_date(context, seg.get("video_date", ""))
                    if context_date[2] != "video_upload_date":
                        event_date, event_date_text, date_source = context_date
                historical_without_exact_date = (
                    evidence_class == "personal_trade"
                    and date_source == "video_upload_date"
                    and re.search(
                        r"\b(?:back in 20\d{2}|in 20\d{2}|years? ago|originally bought|used to buy|was buying)\b",
                        local_context,
                        flags=re.IGNORECASE,
                    )
                )
                if historical_without_exact_date:
                    evidence_class = "historical_personal_trade"
                    strict_eligible = False
                    conf = 0.88
                event_type = {
                    "buy_or_add": "buy_or_add",
                    "sell_or_reduce": "sell_or_reduce",
                    "hold_or_portfolio_holding": "holding_update",
                    "watchlist_or_uncertain": "watchlist",
                    "negative_or_not_buying": "negative_not_buying",
                }.get(action, "mention_only")
                rows.append(
                    {
                        "video_id": video_id,
                        "video_date": seg.get("video_date", ""),
                        "event_date": event_date,
                        "event_date_text": event_date_text,
                        "date_source": date_source,
                        "video_title": seg.get("video_title", ""),
                        "video_url": seg.get("video_url", f"https://www.youtube.com/watch?v={video_id}"),
                        "timestamp_seconds": int(quote_start),
                        "timestamp_start_seconds": int(quote_start),
                        "timestamp_end_seconds": int(quote_end),
                        "timestamp_start": seconds_to_hhmmss(quote_start),
                        "timestamp_end": seconds_to_hhmmss(quote_end),
                        "timestamp_url": f"https://www.youtube.com/watch?v={video_id}&t={int(quote_start)}s",
                        "ticker": ticker,
                        "company": company,
                        "action_inferred": action,
                        "event_type": event_type,
                        "evidence_class": evidence_class,
                        "strict_eligible": strict_eligible,
                        "confidence_rule_based": round(conf, 3),
                        "source_method": seg.get("source_method", "saved_transcript"),
                        "quote_segment": text,
                        "asset_context": local_context,
                        "context_window": context,
                        "manual_review": "spot_check" if strict_eligible else "yes",
                        "verified_action": "",
                        "include_in_portfolio": "",
                        "notes": "",
                    }
                )

    df = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if df.empty:
        return df

    # De-duplicate overlapping repeated mentions of same ticker within the same minute/video.
    df["minute_bucket"] = (df["timestamp_start_seconds"] // 60).astype(int)
    df = df.sort_values(["video_date", "video_title", "timestamp_start_seconds", "ticker"])
    df = df.drop_duplicates(
        subset=["video_url", "ticker", "minute_bucket", "action_inferred", "event_date"]
    )
    df = df.drop(columns=["minute_bucket"])

    return df


def make_review_template(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [
        "video_id",
        "video_date",
        "event_date",
        "event_date_text",
        "date_source",
        "video_title",
        "timestamp_seconds",
        "timestamp_url",
        "ticker",
        "company",
        "action_inferred",
        "event_type",
        "evidence_class",
        "strict_eligible",
        "confidence_rule_based",
        "source_method",
        "quote_segment",
        "asset_context",
        "context_window",
        "manual_review",
        "verified_action",
        "include_in_portfolio",
        "notes",
    ]
    return df[keep].copy()


def create_charts(df: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    plot_df = df.copy()
    plot_df["video_date"] = pd.to_datetime(plot_df["video_date"], errors="coerce")

    top = plot_df["ticker"].value_counts().head(15).sort_values()
    if not top.empty:
        plt.figure(figsize=(9, 6))
        top.plot(kind="barh")
        plt.title("Top detected tickers in transcripts")
        plt.xlabel("Detected candidate mentions")
        plt.tight_layout()
        plt.savefig(charts_dir / "picks_by_ticker.png", dpi=180)
        plt.close()

    actions = plot_df["action_inferred"].value_counts().sort_values()
    if not actions.empty:
        plt.figure(figsize=(9, 5))
        actions.plot(kind="barh")
        plt.title("Candidate actions inferred from transcript context")
        plt.xlabel("Rows")
        plt.tight_layout()
        plt.savefig(charts_dir / "actions_distribution.png", dpi=180)
        plt.close()

    by_month = (
        plot_df.dropna(subset=["video_date"])
        .assign(month=lambda x: x["video_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .size()
    )
    if not by_month.empty:
        plt.figure(figsize=(10, 5))
        by_month.plot(kind="line", marker="o")
        plt.title("Detected candidate stock-pick rows over time")
        plt.xlabel("Month")
        plt.ylabel("Candidate rows")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(charts_dir / "picks_over_time.png", dpi=180)
        plt.close()


def write_summary(inventory: pd.DataFrame, status: pd.DataFrame, candidates: pd.DataFrame, out_path: Path) -> None:
    ok = int((status["transcript_status"].isin(["ok", "already_exists"])).sum()) if not status.empty else 0
    failed = int((status["transcript_status"] == "failed").sum()) if not status.empty else 0

    if candidates.empty:
        top_tickers = "No candidates found."
        action_counts = "No candidates found."
    else:
        top_tickers = candidates["ticker"].value_counts().head(10).to_string()
        action_counts = candidates["action_inferred"].value_counts().to_string()

    text = f"""# Pipeline summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Video collection

- Videos in inventory: {len(inventory)}
- Transcripts downloaded/existing: {ok}
- Transcript failures: {failed}

## Candidate extraction

- Candidate rows: {0 if candidates.empty else len(candidates)}
- Unique tickers: {0 if candidates.empty else candidates['ticker'].nunique()}

### Top tickers

```text
{top_tickers}
```

### Inferred action counts

```text
{action_counts}
```

## Interpretation warning

The CSV is a candidate extraction table, not the final portfolio yet. For the report, manually verify the rows that become portfolio entries/exits. The strongest evidence is a direct portfolio screenshot/update or explicit language such as "I bought", "I sold", "I added", "this is my current holding", or an explicit allocation/weight.

Suggested next step:

1. Open `data/processed/stock_pick_candidates_review_template.csv`.
2. Filter for `buy_or_add`, `sell_or_reduce`, and `hold_or_portfolio_holding`.
3. Watch/check the timestamp URLs.
4. Fill `verified_action` and `include_in_portfolio`.
5. Convert verified rows into the portfolio timeline.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="https://www.youtube.com/@JosephCarlsonShow/videos")
    ap.add_argument("--max-videos", type=int, default=80)
    ap.add_argument("--skip-download", action="store_true", help="Use existing data/video_inventory.csv and transcripts.")
    ap.add_argument("--root", default=".", help="Directory where data and outputs are written.")
    ap.add_argument(
        "--cookies-from-browser",
        choices=["chrome", "edge", "firefox", "brave", "opera", "vivaldi"],
        help="Optional browser session used by the yt-dlp subtitle fallback.",
    )
    ap.add_argument(
        "--request-delay",
        type=float,
        default=0.75,
        help="Seconds to wait between transcript requests.",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    data_dir = root / "data"
    transcript_dir = data_dir / "transcripts"
    processed_dir = data_dir / "processed"
    charts_dir = root / "outputs" / "charts"

    inventory_csv = data_dir / "video_inventory.csv"
    transcript_status_csv = data_dir / "transcript_status.csv"
    transcript_zip = transcript_dir / "transcripts_bundle.zip"
    candidates_csv = processed_dir / "stock_pick_candidates.csv"
    review_csv = processed_dir / "stock_pick_candidates_review_template.csv"
    summary_md = root / "outputs" / "summary_report.md"

    if args.skip_download and inventory_csv.exists():
        inventory = pd.read_csv(inventory_csv)
    else:
        print("Collecting video inventory...")
        inventory = get_video_inventory(args.channel, args.max_videos, inventory_csv)

    if args.skip_download and transcript_status_csv.exists():
        status = pd.read_csv(transcript_status_csv)
    else:
        status = save_transcripts(
            inventory,
            transcript_dir,
            cookies_from_browser=args.cookies_from_browser,
            request_delay=max(args.request_delay, 0.0),
        )
        status.to_csv(transcript_status_csv, index=False)

    print("Bundling transcripts...")
    zip_transcripts(transcript_dir, transcript_zip)

    print("Extracting candidate stock picks / portfolio updates...")
    candidates = extract_candidates(transcript_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_csv, index=False)

    review = make_review_template(candidates)
    review.to_csv(review_csv, index=False)

    print("Creating charts...")
    create_charts(candidates, charts_dir)

    print("Writing summary...")
    write_summary(inventory, status, candidates, summary_md)

    print("\nDONE")
    print(f"Inventory:      {inventory_csv}")
    print(f"Transcript zip: {transcript_zip}")
    print(f"Candidates:     {candidates_csv}")
    print(f"Review file:    {review_csv}")
    print(f"Charts:         {charts_dir}")
    print(f"Summary:        {summary_md}")


if __name__ == "__main__":
    main()
