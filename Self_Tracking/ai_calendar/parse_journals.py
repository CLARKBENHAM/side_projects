"""Parse journal markdown exports into dated structured records."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data"
JOURNAL_FILES = [
    DATA_DIR / "Journal 01_22-10_25.md",
    DATA_DIR / "Journal 11_25- .md",
]
OUTPUT_DIR = BASE_DIR / "journal_outputs"

YEAR_RE = re.compile(r"^##\s+(\d{4})\s*$")
MONTH_RE = re.compile(
    r"^####\s+("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s*(\d{4})?\s*$"
)
DATE_PREFIX_RE = re.compile(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s*(.*)$")
WEEKLY_ITEM_RE = re.compile(r"^\s*(\d+)\.\s+(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\s*(.*)$")
WEEKLY_HEADER_RE = re.compile(r"^####\s+Weekly Summaries", re.IGNORECASE)
REVIEW_HEADER_RE = re.compile(r"^\*+\s*(Quarterly Review|Annual Review)", re.IGNORECASE)
IMAGE_FOOTNOTE_RE = re.compile(r"^\[[^\]]+\]:\s*<data:image/", re.IGNORECASE)

MONTH_TO_NUM = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


@dataclass
class JournalRecord:
    entry_id: str
    source_file: str
    entry_type: str
    date: str
    year_context: int | None
    month_context: int | None
    title: str
    text: str
    week_index: int | None = None


def normalize_year(raw_year: str, current_year: int | None) -> int | None:
    if raw_year:
        year = int(raw_year)
        return year + 2000 if year < 100 else year
    return current_year


def infer_weekly_date(
    month: int,
    day: int,
    explicit_year: str | None,
    current_year: int | None,
    previous_date: pd.Timestamp | None,
) -> pd.Timestamp | None:
    if explicit_year:
        year = normalize_year(explicit_year, current_year)
        return pd.Timestamp(year=year, month=month, day=day)

    if current_year is None:
        return None

    candidates = []
    for year in (current_year - 1, current_year, current_year + 1):
        try:
            candidates.append(pd.Timestamp(year=year, month=month, day=day))
        except ValueError:
            continue

    if not candidates:
        return None

    if previous_date is None:
        if month >= 11:
            for candidate in candidates:
                if candidate.year == current_year - 1:
                    return candidate
        return min(candidates, key=lambda candidate: abs(candidate.year - current_year))

    nondecreasing = [
        candidate for candidate in candidates if candidate >= previous_date
    ]
    if nondecreasing:
        target = previous_date + pd.Timedelta(days=7)
        return min(nondecreasing, key=lambda candidate: abs(candidate - target))

    return max(candidates)


def infer_daily_date(
    month: int,
    day: int,
    explicit_year: str | None,
    current_year: int | None,
    current_month: int | None,
) -> pd.Timestamp | None:
    if explicit_year:
        year = normalize_year(explicit_year, current_year)
    elif current_year is None:
        return None
    else:
        year = current_year
        if current_month is not None:
            if current_month <= 2 and month >= 11:
                year -= 1
            elif current_month >= 11 and month <= 2:
                year += 1

    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def finalize_record(records: list[JournalRecord], current: dict | None) -> None:
    if not current:
        return
    text = "\n".join(current["lines"]).strip()
    if not text:
        return
    records.append(
        JournalRecord(
            entry_id=current["entry_id"],
            source_file=current["source_file"],
            entry_type=current["entry_type"],
            date=(
                current["date"].date().isoformat()
                if current["date"] is not None
                else ""
            ),
            year_context=current["year_context"],
            month_context=current["month_context"],
            title=current["title"].strip(),
            text=text,
            week_index=current.get("week_index"),
        )
    )


def parse_file(path: Path) -> list[JournalRecord]:
    lines = path.read_text().splitlines()
    records: list[JournalRecord] = []
    current_year: int | None = None
    current_month: int | None = None
    in_weekly = False
    in_review_block = False
    current_record: dict | None = None
    prev_daily_date: pd.Timestamp | None = None
    prev_weekly_date: pd.Timestamp | None = None
    seq = 0

    def start_record(
        *,
        entry_type: str,
        date: pd.Timestamp | None,
        title: str,
        line_text: str,
        week_index: int | None = None,
    ) -> None:
        nonlocal current_record, seq, prev_daily_date, prev_weekly_date
        finalize_record(records, current_record)
        seq += 1
        current_record = {
            "entry_id": f"{path.stem.replace(' ', '_').lower()}_{entry_type}_{seq:05d}",
            "source_file": str(path),
            "entry_type": entry_type,
            "date": date,
            "year_context": current_year,
            "month_context": current_month,
            "title": title,
            "lines": [line_text] if line_text else [],
            "week_index": week_index,
        }
        if date is not None:
            if entry_type == "weekly_summary":
                prev_weekly_date = date
            elif entry_type == "daily_entry":
                prev_daily_date = date

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if IMAGE_FOOTNOTE_RE.match(stripped):
            continue

        year_match = YEAR_RE.match(stripped)
        if year_match:
            finalize_record(records, current_record)
            current_record = None
            current_year = int(year_match.group(1))
            current_month = None
            in_weekly = False
            in_review_block = False
            prev_daily_date = None
            prev_weekly_date = None
            continue

        if WEEKLY_HEADER_RE.match(stripped):
            finalize_record(records, current_record)
            current_record = None
            in_weekly = True
            in_review_block = False
            prev_weekly_date = None
            continue

        if REVIEW_HEADER_RE.match(stripped):
            finalize_record(records, current_record)
            current_record = None
            in_review_block = True
            continue

        month_match = MONTH_RE.match(stripped)
        if month_match:
            finalize_record(records, current_record)
            current_record = None
            current_month = MONTH_TO_NUM[month_match.group(1)]
            if month_match.group(2):
                current_year = int(month_match.group(2))
            in_weekly = False
            in_review_block = False
            prev_daily_date = None
            continue

        if stripped.startswith("## ") or stripped.startswith("#### "):
            finalize_record(records, current_record)
            current_record = None
            if stripped.startswith("#### "):
                in_weekly = False
                in_review_block = False
            continue

        if in_weekly:
            weekly_match = WEEKLY_ITEM_RE.match(line)
            if weekly_match:
                month = int(weekly_match.group(2))
                day = int(weekly_match.group(3))
                date = infer_weekly_date(
                    month,
                    day,
                    weekly_match.group(4),
                    current_year,
                    prev_weekly_date,
                )
                start_record(
                    entry_type="weekly_summary",
                    date=date,
                    title=weekly_match.group(5),
                    line_text="" if in_review_block else weekly_match.group(5),
                    week_index=int(weekly_match.group(1)),
                )
                continue
            if current_record is not None and not in_review_block:
                current_record["lines"].append(line)
            continue

        if in_review_block:
            continue

        date_match = DATE_PREFIX_RE.match(stripped)
        if date_match:
            month = int(date_match.group(1))
            day = int(date_match.group(2))
            date = infer_daily_date(
                month,
                day,
                date_match.group(3),
                current_year,
                current_month,
            )
            remainder = date_match.group(4).strip()
            start_record(
                entry_type="daily_entry",
                date=date,
                title=remainder,
                line_text=remainder,
            )
            continue

        if current_record is not None:
            current_record["lines"].append(line)

    finalize_record(records, current_record)
    return records


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[JournalRecord] = []
    for journal_file in JOURNAL_FILES:
        records.extend(parse_file(journal_file))

    df = pd.DataFrame(asdict(record) for record in records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "entry_type", "entry_id"]).reset_index(drop=True)
    df["text_len"] = df["text"].str.len()
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")

    df.to_csv(OUTPUT_DIR / "journal_entries.csv", index=False)
    df[df["entry_type"] == "weekly_summary"].to_csv(
        OUTPUT_DIR / "journal_weekly_summaries.csv", index=False
    )
    df[df["entry_type"] == "daily_entry"].to_csv(
        OUTPUT_DIR / "journal_daily_entries.csv", index=False
    )

    print(f"Parsed records: {len(df)}")
    print(df["entry_type"].value_counts().to_string())
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print("\nTop 10 longest entries:")
    cols = ["date", "entry_type", "title", "text_len"]
    print(df.nlargest(10, "text_len")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
