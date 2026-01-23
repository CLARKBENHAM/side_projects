#!/usr/bin/env python3
"""
Aggregate LeetCode analysis outputs and send to Claude.

Usage:
  python3 interview_practice/leetcode_aggregate.py \
    --slugs-file interview_practice/leetcode_analysis/leetcode_assesment_slugs.txt \
    --analysis-dir interview_practice/leetcode_analysis \
    --cli claude
"""

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Entry:
    index: int
    slug: str
    run_dir: str
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate LeetCode analyses.")
    parser.add_argument(
        "--slugs-file",
        required=True,
        help="Path to slugs file (one slug per line)",
    )
    parser.add_argument(
        "--analysis-dir",
        default=os.path.join("interview_practice", "leetcode_analysis"),
        help="Base analysis directory",
    )
    parser.add_argument(
        "--cli",
        default="claude",
        help="CLI command to run (default: claude)",
    )
    parser.add_argument(
        "--cli-args",
        default="",
        help="Extra args passed to CLI, quoted as a single string",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Max characters per chunk sent to Claude",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output file path for Claude response",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build prompt and write output files",
    )
    return parser.parse_args()


def read_slugs(slugs_file: str) -> list[str]:
    slugs = []
    with open(slugs_file, "r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if cleaned and not cleaned.startswith("#"):
                slugs.append(cleaned)
    return slugs


def find_latest_cli_output(analysis_dir: str, slug: str) -> tuple[str, str] | None:
    if not os.path.isdir(analysis_dir):
        return None
    candidates = []
    for entry in os.listdir(analysis_dir):
        if entry.startswith(f"{slug}_"):
            run_dir = os.path.join(analysis_dir, entry)
            cli_path = os.path.join(run_dir, "cli_output.txt")
            if os.path.isfile(cli_path):
                candidates.append((run_dir, cli_path))
    candidates.sort(key=lambda t: os.path.getmtime(t[0]), reverse=True)
    for run_dir, cli_path in candidates:
        try:
            with open(cli_path, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
            if not content:
                continue
            if "Skipping LLM call" in content:
                continue
            return run_dir, content
        except OSError:
            continue
    return None


def extract_analysis_text(content: str) -> str:
    cleaned = content.strip()
    if not cleaned:
        return ""
    if "\n\n[stderr]\n" in cleaned:
        cleaned = cleaned.split("\n\n[stderr]\n", 1)[0].strip()
    if cleaned.startswith("$ "):
        lines = cleaned.splitlines()
        if len(lines) > 1:
            cleaned = "\n".join(lines[1:]).strip()
    return cleaned


def build_prompt(entries: list[Entry]) -> str:
    header = [
        "You are analyzing a sequence of LeetCode postmortems.",
        "Each item includes the slug order index from the assessment slug file.",
        "Use the order index to infer recency (higher index is newer).",
        "",
        "Please categorize recurring mistakes, improvements over time,",
        "and give a ranked action plan for what to practice next.",
        "",
    ]
    body = []
    for entry in entries:
        body.append(f"[{entry.index}] {entry.slug}")
        body.append(entry.content.strip())
        body.append("")
    return "\n".join(header + body).strip() + "\n"


def chunk_entries(entries: list[Entry], max_chars: int) -> list[list[Entry]]:
    chunks: list[list[Entry]] = []
    current: list[Entry] = []
    current_len = 0
    for entry in entries:
        snippet = f"[{entry.index}] {entry.slug}\n{entry.content}\n\n"
        if current and current_len + len(snippet) > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(entry)
        current_len += len(snippet)
    if current:
        chunks.append(current)
    return chunks


def run_cli(prompt: str, cli_command: str, cli_args: list[str]) -> str:
    proc = subprocess.run(
        [cli_command, *cli_args],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    output = []
    output.append(f"$ {' '.join([cli_command, *cli_args])}")
    if proc.stdout:
        output.append(proc.stdout)
    if proc.stderr:
        output.append("\n[stderr]\n" + proc.stderr)
    if proc.returncode != 0:
        output.append(f"\n[exit code {proc.returncode}]")
    return "\n".join(output).strip() + "\n"


def main() -> None:
    args = parse_args()
    slugs = read_slugs(args.slugs_file)
    analysis_dir = os.path.abspath(args.analysis_dir)

    entries: list[Entry] = []
    for idx, slug in enumerate(slugs, start=1):
        found = find_latest_cli_output(analysis_dir, slug)
        if not found:
            continue
        run_dir, content = found
        analysis_text = extract_analysis_text(content)
        if not analysis_text:
            continue
        entries.append(Entry(index=idx, slug=slug, run_dir=run_dir, content=analysis_text))

    if not entries:
        raise SystemExit("No non-empty cli_output.txt files found.")

    chunks = chunk_entries(entries, max_chars=max(1000, args.max_chars))
    prompts = [build_prompt(chunk) for chunk in chunks]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_out = os.path.join(analysis_dir, f"aggregate_summary_{timestamp}.txt")
    out_path = args.out or default_out

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for i, prompt in enumerate(prompts, start=1):
            handle.write(f"PROMPT CHUNK {i}/{len(prompts)}\n")
            handle.write(prompt)
            handle.write("\n\n")

    if args.dry_run:
        return

    if not shutil.which(args.cli):
        raise SystemExit(f"CLI not found: {args.cli}")

    cli_args = args.cli_args.split() if args.cli_args else []
    chunk_outputs = []
    for i, prompt in enumerate(prompts, start=1):
        cli_output = run_cli(prompt, args.cli, cli_args)
        chunk_outputs.append(cli_output)
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write(f"RESPONSE CHUNK {i}/{len(prompts)}\n")
            handle.write(cli_output)
            handle.write("\n\n")

    if len(chunk_outputs) > 1:
        final_prompt = (
            "You are summarizing multiple chunk analyses of LeetCode postmortems.\n"
            "Combine them into a single ranked action plan and trend summary.\n\n"
            + "\n\n".join(chunk_outputs)
        )
        final_output = run_cli(final_prompt, args.cli, cli_args)
        with open(out_path, "a", encoding="utf-8") as handle:
            handle.write("FINAL RESPONSE\n")
            handle.write(final_output)


if __name__ == "__main__":
    main()
