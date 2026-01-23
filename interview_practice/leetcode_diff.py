#!/usr/bin/env python3
"""
Fetch all submissions for a LeetCode problem and generate diffs.

Usage:
    1. Get your LEETCODE_SESSION cookie from browser DevTools
    2. python leetcode_diff.py <problem-slug> <session-cookie>

Example:
    python leetcode_diff.py minimum-pair-removal-to-sort-array-ii "eyJ0eXAiOi..."
From File:
export LEETCODE_SESSION="eyJ0eXAiOi..."
python3 interview_practice/leetcode_diff.py --slugs-file interview_practice/leetcode_analysis/leetcode_assesment_slugs.txt --workers 3

Pulled problem statement and latest failing testcase, wrote diff-only output to a txt file, and ran Codex non‑interactively. The full run artifacts are now in a fixed folder:
interview_practice/leetcode_analysis/minimum-pair-removal-to-sort-array-ii_20260123_114134/
diffs.txt (diff‑only output)
prompt.txt (problem + diffs + failing testcase + notes placeholder)
submissions.json (full structured output)
cli_output.txt (Codex answer + prompt)
Codex ran via codex exec - and produced the analysis; it’s saved in cli_output.txt.
What I changed in leetcode_diff.py
Fetches problem statement via GraphQL (questionContent)
Fetches lastTestcase + expectedOutput where available
Writes diff‑only text to diffs.txt
Writes prompt + CLI output into a timestamped subfolder
Adds --notes-file so you can include self‑analysis notes in the prompt
Runs codex exec - by default when codex is installed
Example usage (with notes):
python3 interview_practice/leetcode_diff.py minimum-pair-removal-to-sort-array-ii <LEETCODE_SESSION> --notes-file /path/to/notes.txt

If you want Claude instead:
python3 interview_practice/leetcode_diff.py <slug> <LEETCODE_SESSION> --cli claude --cli-args "<args>"

"""

import argparse
import difflib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import requests

GRAPHQL_URL = "https://leetcode.com/graphql"

SUBMISSION_LIST_QUERY = """
query submissionList($offset: Int!, $limit: Int!, $questionSlug: String!) {
    submissionList(offset: $offset, limit: $limit, questionSlug: $questionSlug) {
        lastKey
        hasNext
        submissions {
            id
            statusDisplay
            lang
            runtime
            timestamp
            memory
        }
    }
}
"""

SUBMISSION_DETAIL_QUERY = """
query submissionDetails($submissionId: Int!) {
    submissionDetails(submissionId: $submissionId) {
        code
    }
}
"""

SUBMISSION_DETAIL_WITH_TESTCASE_QUERY = """
query submissionDetails($submissionId: Int!) {
    submissionDetails(submissionId: $submissionId) {
        code
        lastTestcase
        expectedOutput
    }
}
"""

QUESTION_CONTENT_QUERY = """
query questionContent($titleSlug: String!) {
    question(titleSlug: $titleSlug) {
        title
        content
    }
}
"""

DEFAULT_OUTPUT_DIR = os.path.join("interview_practice", "leetcode_analysis")


@dataclass
class RunPaths:
    run_dir: str
    json_path: str
    diffs_path: str
    prompt_path: str
    cli_output_path: str


def build_session(session_cookie: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Origin": "https://leetcode.com",
            "Referer": "https://leetcode.com",
            "User-Agent": "Mozilla/5.0",
        }
    )
    session.cookies.set("LEETCODE_SESSION", session_cookie, domain=".leetcode.com")
    # Fetch CSRF token cookie required by LeetCode GraphQL.
    session.get("https://leetcode.com")
    return session


def graphql_request(session: requests.Session, query: str, variables: dict) -> dict:
    csrftoken = session.cookies.get("csrftoken")
    headers = {}
    if csrftoken:
        headers["x-csrftoken"] = csrftoken
    response = session.post(
        GRAPHQL_URL,
        headers=headers,
        json={"query": query, "variables": variables},
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code} for GraphQL: {response.text}")
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL error: {payload['errors']}")
    return payload


def fetch_all_submissions(session: requests.Session, slug: str) -> list[dict]:
    """Fetch all submission metadata for a problem."""
    all_submissions = []
    offset = 0
    limit = 20

    while True:
        result = graphql_request(
            session,
            SUBMISSION_LIST_QUERY,
            {"offset": offset, "limit": limit, "questionSlug": slug},
        )
        submission_list = result.get("data", {}).get("submissionList")
        if submission_list is None:
            raise RuntimeError(f"submissionList missing in response: {result}")
        submissions = submission_list.get("submissions") or []
        all_submissions.extend(submissions)

        if not submission_list.get("hasNext"):
            break
        offset += limit

    return all_submissions


def fetch_all_submissions_rest(session: requests.Session, slug: str) -> list[dict]:
    all_submissions = []
    offset = 0
    limit = 20
    last_key = None
    while True:
        params = {"offset": offset, "limit": limit}
        if last_key:
            params["last_key"] = last_key
        response = session.get(f"https://leetcode.com/api/submissions/{slug}/", params=params)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code} for REST submissions: {response.text}")
        payload = response.json()
        submissions = payload.get("submissions_dump") or []
        for sub in submissions:
            all_submissions.append(
                {
                    "id": str(sub.get("id")),
                    "statusDisplay": sub.get("status_display"),
                    "lang": sub.get("lang"),
                    "runtime": sub.get("runtime"),
                    "timestamp": str(sub.get("timestamp")),
                    "memory": sub.get("memory"),
                    "code": sub.get("code", ""),
                }
            )
        if not payload.get("has_next"):
            break
        last_key = payload.get("last_key")
        offset += limit
    return all_submissions


def fetch_submission_details(session: requests.Session, submission_id: int) -> dict:
    """Fetch the actual code for a submission (and testcase if available)."""
    last_error = None
    for query in (SUBMISSION_DETAIL_WITH_TESTCASE_QUERY, SUBMISSION_DETAIL_QUERY):
        try:
            result = graphql_request(session, query, {"submissionId": submission_id})
            details = result.get("data", {}).get("submissionDetails")
            if details is None:
                raise RuntimeError(f"submissionDetails missing in response: {result}")
            return details
        except RuntimeError as exc:
            last_error = exc
    raise last_error


def fetch_problem_statement(session: requests.Session, slug: str) -> dict:
    result = graphql_request(session, QUESTION_CONTENT_QUERY, {"titleSlug": slug})
    question = result.get("data", {}).get("question")
    if question is None:
        raise RuntimeError(f"question missing in response: {result}")
    return question


def format_timestamp(ts: str) -> str:
    return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def generate_diff(old_code: str, new_code: str, old_meta: dict, new_meta: dict) -> str:
    """Generate unified diff between two submissions."""
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    old_label = (
        f"submission_{old_meta['id']} ({old_meta['statusDisplay']}) @"
        f" {format_timestamp(old_meta['timestamp'])}"
    )
    new_label = (
        f"submission_{new_meta['id']} ({new_meta['statusDisplay']}) @"
        f" {format_timestamp(new_meta['timestamp'])}"
    )

    diff = difflib.unified_diff(old_lines, new_lines, fromfile=old_label, tofile=new_label)
    return "".join(diff)


def resolve_cli_command(cli_command: str | None) -> str | None:
    if cli_command:
        return cli_command
    for candidate in ("codex", "claude"):
        if shutil.which(candidate):
            return candidate
    return None


def build_run_paths(base_dir: str, slug: str) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{slug}_{timestamp}")
    return RunPaths(
        run_dir=run_dir,
        json_path=os.path.join(run_dir, "submissions.json"),
        diffs_path=os.path.join(run_dir, "diffs.txt"),
        prompt_path=os.path.join(run_dir, "prompt.txt"),
        cli_output_path=os.path.join(run_dir, "cli_output.txt"),
    )


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def load_cached_output(base_dir: str, slug: str) -> dict | None:
    if not os.path.isdir(base_dir):
        return None
    candidates = []
    for entry in os.listdir(base_dir):
        if entry.startswith(f"{slug}_"):
            path = os.path.join(base_dir, entry)
            candidates.append(path)
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for run_dir in candidates:
        submissions_path = os.path.join(run_dir, "submissions.json")
        if not os.path.isfile(submissions_path):
            continue
        try:
            with open(submissions_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("total_submissions", 0) > 0 and data.get("submissions"):
                return data
        except (OSError, json.JSONDecodeError):
            continue
    return None


def should_skip_llm_for_prompt(base_dir: str, slug: str, prompt: str) -> bool:
    if not os.path.isdir(base_dir):
        return False
    for entry in os.listdir(base_dir):
        if not entry.startswith(f"{slug}_"):
            continue
        run_dir = os.path.join(base_dir, entry)
        prompt_path = os.path.join(run_dir, "prompt.txt")
        cli_path = os.path.join(run_dir, "cli_output.txt")
        if not (os.path.isfile(prompt_path) and os.path.isfile(cli_path)):
            continue
        try:
            with open(prompt_path, "r", encoding="utf-8") as handle:
                prior_prompt = handle.read()
            if prior_prompt != prompt:
                continue
            with open(cli_path, "r", encoding="utf-8") as handle:
                cli_output = handle.read()
            if "Skipping LLM call" in cli_output:
                continue
            return True
        except OSError:
            continue
    return False


def run_cli(prompt: str, cli_command: str, cli_args: list[str]) -> str:
    effective_args = list(cli_args)
    if cli_command == "codex" and not effective_args:
        effective_args = ["exec", "-"]
    proc = subprocess.run(
        [cli_command, *effective_args],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    output = []
    output.append(f"$ {' '.join([cli_command, *effective_args])}")
    if proc.stdout:
        output.append(proc.stdout)
    if proc.stderr:
        output.append("\n[stderr]\n" + proc.stderr)
    if proc.returncode != 0:
        output.append(f"\n[exit code {proc.returncode}]")
    return "\n".join(output).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch LeetCode submissions and diffs.")
    parser.add_argument("slug", nargs="?", help="LeetCode problem slug, e.g. two-sum")
    parser.add_argument(
        "session_cookie",
        nargs="?",
        default="",
        help="Value of LEETCODE_SESSION cookie (or set LEETCODE_SESSION env var)",
    )
    parser.add_argument(
        "--slugs",
        default="",
        help="Comma-separated list of slugs to process",
    )
    parser.add_argument(
        "--slugs-file",
        default="",
        help="Path to a file containing slugs (one per line)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max concurrent workers for multiple slugs",
    )
    parser.add_argument("--lang", dest="lang_filter", help="Filter by language, e.g. python3")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, help="Base output directory")
    parser.add_argument(
        "--cli", dest="cli_command", default=None, help="CLI command: codex or claude"
    )
    parser.add_argument(
        "--cli-args",
        default="",
        help="Extra args passed to the CLI, quoted as a single string",
    )
    parser.add_argument(
        "--notes-file",
        default="",
        help="Optional path to a text file with your self-analysis notes",
    )
    parser.add_argument(
        "--always-run-llm",
        action="store_true",
        help="Run the LLM even if all submissions are accepted",
    )
    return parser.parse_args()


def load_slugs(args: argparse.Namespace) -> list[str]:
    slugs = []
    if args.slug:
        slugs.append(args.slug)
    if args.slugs:
        slugs.extend([s.strip() for s in args.slugs.split(",") if s.strip()])
    if args.slugs_file:
        with open(args.slugs_file, "r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    slugs.append(cleaned)
    deduped = []
    seen = set()
    for slug in slugs:
        if slug not in seen:
            deduped.append(slug)
            seen.add(slug)
    return deduped


def process_slug(
    slug: str,
    session_cookie: str,
    lang_filter: str | None,
    out_dir: str,
    notes_text: str,
) -> dict:
    print(f"Fetching submissions for: {slug}", file=sys.stderr)

    session = build_session(session_cookie)
    problem = fetch_problem_statement(session, slug)
    base_dir = os.path.abspath(out_dir)
    submissions = fetch_all_submissions(session, slug)
    if not submissions:
        rest_submissions = fetch_all_submissions_rest(session, slug)
        if rest_submissions:
            submissions = rest_submissions

    if lang_filter:
        submissions = [s for s in submissions if s["lang"] == lang_filter]

    submissions.sort(key=lambda x: int(x["timestamp"]))

    print(f"Found {len(submissions)} submissions", file=sys.stderr)

    if not submissions:
        cached_output = load_cached_output(base_dir, slug)
        if cached_output:
            output = cached_output
            output["problem_title"] = output.get("problem_title") or problem.get("title")
            output["problem_statement"] = output.get("problem_statement") or problem.get("content")
            run_paths = build_run_paths(base_dir, slug)
            diffs_text = "\n\n".join(d["diff"] for d in output.get("diffs", []))
            write_text(run_paths.diffs_path, diffs_text + ("\n" if diffs_text else ""))
            write_text(run_paths.json_path, json.dumps(output, indent=2))
            prompt = PROMPT_TEMPLATE.format(
                problem_statement=output.get("problem_statement", ""),
                diffs_json=json.dumps(output.get("diffs", []), indent=2),
                failing_testcase=json.dumps(
                    {
                        "submission_id": None,
                        "status": None,
                        "last_testcase": None,
                        "expected_output": None,
                    },
                    indent=2,
                ),
                notes=notes_text,
            )
            write_text(run_paths.prompt_path, prompt)
            return {
                "output": output,
                "prompt": prompt,
                "run_paths": run_paths,
                "has_incorrect": True,
            }
        output = {
            "problem_slug": slug,
            "problem_title": problem.get("title"),
            "problem_statement": problem.get("content"),
            "total_submissions": 0,
            "submissions": [],
            "diffs": [],
        }
        run_paths = build_run_paths(base_dir, slug)
        write_text(run_paths.diffs_path, "")
        write_text(run_paths.json_path, json.dumps(output, indent=2))
        prompt = PROMPT_TEMPLATE.format(
            problem_statement=problem.get("content", ""),
            diffs_json="[]",
            failing_testcase=json.dumps(
                {
                    "submission_id": None,
                    "status": None,
                    "last_testcase": None,
                    "expected_output": None,
                },
                indent=2,
            ),
            notes=notes_text,
        )
        write_text(run_paths.prompt_path, prompt)
        return {
            "output": output,
            "prompt": prompt,
            "run_paths": run_paths,
            "has_incorrect": False,
        }

    codes = []
    rest_code_map = None
    for i, sub in enumerate(submissions):
        detail = {}
        code = sub.get("code")
        if code is None:
            if i == 0:
                print(f"Fetching code for {len(submissions)} submissions...", file=sys.stderr)
            print(f"  [{i+1}/{len(submissions)}] Fetching submission {sub['id']}...", file=sys.stderr)
            try:
                detail = fetch_submission_details(session, int(sub["id"]))
                code = detail.get("code", "")
            except RuntimeError:
                detail = {}
                code = ""
        if not code:
            if rest_code_map is None:
                rest_code_map = {
                    entry["id"]: entry for entry in fetch_all_submissions_rest(session, slug)
                }
            rest_entry = rest_code_map.get(str(sub["id"]))
            if rest_entry:
                code = rest_entry.get("code", "")
        codes.append(
            {
                **sub,
                "code": code or "",
                "detail": detail,
            }
        )

    output = {
        "problem_slug": slug,
        "problem_title": problem.get("title"),
        "problem_statement": problem.get("content"),
        "total_submissions": len(codes),
        "submissions": [],
        "diffs": [],
    }

    for i, code_data in enumerate(codes):
        output["submissions"].append(
            {
                "index": i,
                "id": code_data["id"],
                "timestamp": format_timestamp(code_data["timestamp"]),
                "status": code_data["statusDisplay"],
                "lang": code_data["lang"],
                "runtime": code_data.get("runtime"),
                "memory": code_data.get("memory"),
                "code": code_data["code"],
                "last_testcase": code_data.get("detail", {}).get("lastTestcase"),
                "expected_output": code_data.get("detail", {}).get("expectedOutput"),
            }
        )

        if i > 0:
            diff = generate_diff(
                codes[i - 1]["code"],
                codes[i]["code"],
                codes[i - 1],
                codes[i],
            )
            output["diffs"].append(
                {
                    "from_index": i - 1,
                    "to_index": i,
                    "diff": diff,
                }
            )

    base_dir = os.path.abspath(out_dir)
    run_paths = build_run_paths(base_dir, slug)
    diffs_text = "\n\n".join(d["diff"] for d in output["diffs"])
    write_text(run_paths.diffs_path, diffs_text + ("\n" if diffs_text else ""))
    write_text(run_paths.json_path, json.dumps(output, indent=2))

    failing_case = None
    failure_statuses = {
        "Wrong Answer",
        "Runtime Error",
        "Time Limit Exceeded",
        "Memory Limit Exceeded",
    }
    for entry in reversed(output["submissions"]):
        if entry["status"] in failure_statuses and not entry.get("last_testcase"):
            try:
                detail = fetch_submission_details(session, int(entry["id"]))
                entry["last_testcase"] = detail.get("lastTestcase")
                entry["expected_output"] = detail.get("expectedOutput")
            except RuntimeError:
                pass
        if entry.get("last_testcase") and entry["status"] in failure_statuses:
            failing_case = {
                "submission_id": entry["id"],
                "status": entry["status"],
                "last_testcase": entry["last_testcase"],
                "expected_output": entry.get("expected_output"),
            }
            break
    if failing_case is None:
        failing_case = {
            "submission_id": None,
            "status": None,
            "last_testcase": None,
            "expected_output": None,
        }

    prompt = PROMPT_TEMPLATE.format(
        problem_statement=problem.get("content", ""),
        diffs_json=json.dumps(output["diffs"], indent=2),
        failing_testcase=json.dumps(failing_case, indent=2),
        notes=notes_text,
    )
    write_text(run_paths.prompt_path, prompt)

    has_incorrect = any(entry["status"] in failure_statuses for entry in output["submissions"])

    return {
        "output": output,
        "prompt": prompt,
        "run_paths": run_paths,
        "has_incorrect": has_incorrect,
    }


def run_llm_for_slug(
    prompt: str,
    run_paths: RunPaths,
    cli_command: str | None,
    cli_args: list[str],
) -> None:
    if cli_command:
        cli_output = run_cli(prompt, cli_command, cli_args)
        write_text(run_paths.cli_output_path, cli_output)
    else:
        write_text(
            run_paths.cli_output_path,
            "Skipping LLM call (no CLI).\n",
        )


def main():
    args = parse_args()
    slugs = load_slugs(args)
    if not slugs:
        raise SystemExit("Provide a slug or --slugs/--slugs-file.")

    session_cookie = args.session_cookie or os.environ.get("LEETCODE_SESSION", "")
    if not session_cookie:
        raise SystemExit("Provide LEETCODE_SESSION as an argument or env var.")

    notes_text = ""
    if args.notes_file:
        with open(args.notes_file, "r", encoding="utf-8") as handle:
            notes_text = handle.read().strip()

    cli_command = resolve_cli_command(args.cli_command)
    cli_args = args.cli_args.split() if args.cli_args else []

    results = []
    llm_jobs = []
    for idx, slug in enumerate(slugs):
        if idx > 0:
            time.sleep(10 + random.random() * 20)
        result = process_slug(
            slug,
            session_cookie,
            args.lang_filter,
            args.out_dir,
            notes_text,
        )
        results.append(result["output"])
        should_run = args.always_run_llm or result["has_incorrect"]
        if should_run:
            base_dir = os.path.abspath(args.out_dir)
            if should_skip_llm_for_prompt(base_dir, slug, result["prompt"]):
                write_text(
                    result["run_paths"].cli_output_path,
                    "Skipping LLM call (cached prompt match).\n",
                )
            else:
                llm_jobs.append((result["prompt"], result["run_paths"]))
        else:
            write_text(
                result["run_paths"].cli_output_path,
                "Skipping LLM call (all submissions accepted).\n",
            )

    if llm_jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(run_llm_for_slug, prompt, run_paths, cli_command, cli_args)
                for prompt, run_paths in llm_jobs
            ]
            for future in as_completed(futures):
                future.result()

    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))


PROMPT_TEMPLATE = """Here is the problem statement:

<problem>
{problem_statement}
</problem>

Here is my submission progression (chronological diffs):

<submissions>
{diffs_json}
</submissions>

Failing testcase (from my latest non-AC submission, if available):
<failing_testcase>
{failing_testcase}
</failing_testcase>

My self-analysis notes (optional):
<notes>
{notes}
</notes>

Please analyze:
1. What patterns of bugs am I making?
2. What's my debugging strategy, and is it effective?
3. What concepts do I seem to struggle with?
4. What specific habits should I change?"""

if __name__ == "__main__":
    main()
