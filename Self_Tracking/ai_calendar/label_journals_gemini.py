"""Label parsed journal entries with Gemini using an API key."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "journal_outputs"
OUTPUT_DIR = INPUT_DIR
DEFAULT_MODEL = "gemini-2.5-flash"
API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

LABEL_FIELDS = {
    "mood_valence": "integer from -2 to 2",
    "energy_activation": "integer from -2 to 2",
    "goal_clarity": "integer from 0 to 3",
    "execution_focus": "integer from 0 to 3",
    "overanalysis": "integer from 0 to 3",
    "avoidance": "integer from 0 to 3",
    "self_criticism": "integer from 0 to 3",
    "relationship_conflict": "integer from 0 to 3",
    "relationship_closeness": "integer from 0 to 3",
    "alcohol_issue": "integer from 0 to 3",
    "porn_issue": "integer from 0 to 3",
    "phone_internet_issue": "integer from 0 to 3",
    "travel_disruption": "integer from 0 to 3",
    "illness_or_pain": "integer from 0 to 3",
    "external_structure": "integer from 0 to 3",
    "accountability_support": "integer from 0 to 3",
    "compelling_problem": "integer from 0 to 3",
    "dominant_obstacle": (
        "one of: relationship_conflict, diffuse_life_load, alcohol, porn, "
        "phone_internet, illness_pain, travel_logistics, overanalysis, "
        "fear_avoidance, low_clarity, low_structure, sleep_circadian, other"
    ),
    "dominant_driver": (
        "one of: compelling_problem, accountability, external_structure, "
        "technical_absorption, hope_vision, relationship_support, routine, "
        "stimulant_support, rest_recovery, other"
    ),
    "dominant_mode": (
        "one of: execution, overanalysis, avoidance, collapse, recovery, "
        "relationship, logistics, reflection"
    ),
    "one_sentence_summary": "short sentence",
}


def build_prompt(batch: list[dict[str, str]]) -> str:
    schema_text = "\n".join(f"- {key}: {value}" for key, value in LABEL_FIELDS.items())
    records_text = []
    for record in batch:
        records_text.append(
            "\n".join(
                [
                    f"ENTRY_ID: {record['entry_id']}",
                    f"ENTRY_TYPE: {record['entry_type']}",
                    f"DATE: {record['date']}",
                    f"TITLE: {record['title']}",
                    "TEXT:",
                    record["text"][:12000],
                ]
            )
        )
    return (
        "You are labeling journal entries for later quantitative analysis.\n"
        "Do not flatter. Do not psychoanalyze beyond the text. Treat the author's "
        "explanations as hypotheses, not facts. Score only what is supported by "
        "the entry.\n\n"
        "Return ONLY valid JSON as an array of objects, one per entry, in the same "
        "order. Each object must contain:\n"
        "- entry_id\n"
        f"{schema_text}\n\n"
        "Scoring rules:\n"
        "- 0 means absent or not supported.\n"
        "- Use higher values only for clear salience.\n"
        "- relationship_closeness and relationship_conflict can both be nonzero.\n"
        "- If uncertain, choose lower intensity.\n\n"
        "Entries:\n\n" + "\n\n---\n\n".join(records_text)
    )


def clean_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def call_gemini(
    session: requests.Session, api_key: str, model: str, prompt: str
) -> list[dict]:
    url = API_URL.format(model=model)
    response = session.post(
        url,
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(clean_json(text))


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    existing = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            existing[record["entry_id"]] = record
    return existing


def append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entry-type", choices=["weekly_summary", "daily_entry"], required=True
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-text-len", type=int, default=6000)
    parser.add_argument("--weekend-only", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")

    input_path = INPUT_DIR / "journal_entries.csv"
    output_path = OUTPUT_DIR / f"{args.entry_type}_labels.jsonl"

    df = pd.read_csv(input_path)
    df = df[df["entry_type"] == args.entry_type].copy()
    df = df[df["text_len"] >= 40].copy()
    df = df[df["text_len"] <= args.max_text_len].copy()
    if args.weekend_only:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].dt.dayofweek >= 5].copy()

    existing = load_existing(output_path)
    df = df[~df["entry_id"].isin(existing)].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f"Need labels: {len(df)}")
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    rows = df.to_dict("records")
    total_done = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompt = build_prompt(batch)
        for attempt in range(4):
            try:
                labels = call_gemini(session, api_key, args.model, prompt)
                if len(labels) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} labels, received {len(labels)}"
                    )
                append_jsonl(output_path, labels)
                total_done += len(labels)
                print(
                    f"Labeled {total_done}/{len(rows)} for {args.entry_type} "
                    f"(last batch size={len(batch)})"
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise
                wait = 2**attempt
                print(f"Retrying after error: {exc} (sleep {wait}s)")
                time.sleep(wait)


if __name__ == "__main__":
    main()
