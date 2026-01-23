#!/usr/bin/env python3
"""
Fetch all submissions for a LeetCode problem and generate diffs.

Usage:
    1. Get your LEETCODE_SESSION cookie from browser DevTools
    2. python leetcode_diff.py <problem-slug> <session-cookie>

Example:
    python leetcode_diff.py minimum-pair-removal-to-sort-array-ii "eyJ0eXAiOi..."

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
import shutil
import subprocess
import sys
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
        submissions = submission_list.get("submissions", [])
        all_submissions.extend(submissions)

        if not submission_list.get("hasNext"):
            break
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
    parser.add_argument("slug", help="LeetCode problem slug, e.g. two-sum")
    parser.add_argument("session_cookie", help="Value of LEETCODE_SESSION cookie")
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
    return parser.parse_args()


def main():
    args = parse_args()
    slug = args.slug
    session_cookie = args.session_cookie
    lang_filter = args.lang_filter

    print(f"Fetching submissions for: {slug}", file=sys.stderr)

    session = build_session(session_cookie)
    problem = fetch_problem_statement(session, slug)
    submissions = fetch_all_submissions(session, slug)

    if lang_filter:
        submissions = [s for s in submissions if s["lang"] == lang_filter]

    submissions.sort(key=lambda x: int(x["timestamp"]))

    print(f"Found {len(submissions)} submissions", file=sys.stderr)

    if not submissions:
        print("No submissions found.")
        sys.exit(0)

    print(f"Fetching code for {len(submissions)} submissions...", file=sys.stderr)

    codes = []
    for i, sub in enumerate(submissions):
        print(f"  [{i+1}/{len(submissions)}] Fetching submission {sub['id']}...", file=sys.stderr)
        detail = fetch_submission_details(session, int(sub["id"]))
        codes.append(
            {
                **sub,
                "code": detail.get("code", ""),
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

    base_dir = os.path.abspath(args.out_dir)
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

    notes_text = ""
    if args.notes_file:
        with open(args.notes_file, "r", encoding="utf-8") as handle:
            notes_text = handle.read().strip()

    prompt = PROMPT_TEMPLATE.format(
        problem_statement=problem.get("content", ""),
        diffs_json=json.dumps(output["diffs"], indent=2),
        failing_testcase=json.dumps(failing_case, indent=2),
        notes=notes_text,
    )
    write_text(run_paths.prompt_path, prompt)

    cli_command = resolve_cli_command(args.cli_command)
    if cli_command:
        cli_args = args.cli_args.split() if args.cli_args else []
        cli_output = run_cli(prompt, cli_command, cli_args)
        write_text(run_paths.cli_output_path, cli_output)
    else:
        write_text(
            run_paths.cli_output_path,
            "No CLI found. Install 'codex' or 'claude' or pass --cli.\n",
        )

    print(json.dumps(output, indent=2))


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
