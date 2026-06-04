# Pre-Read Correctness Review: Master of the Senate

This bundle is for judging whether the new pre-read pipeline is aimed at the
right target before running it across new books.

Files:

- `prompt_contract.md`: what the new prompt is supposed to optimize for.
- `failure_cases.md`: the two concrete misses you called out, with expected behavior.
- `case_01_apparent_power.md`: source-grounded review file for the apparent-power highlight.
- `case_02_rowe_scene.md`: source-grounded review file for the Rowe crying/reassertion scene.
- `review_checklist.md`: pass/fail checklist for generated briefs.
- `run_commands.md`: commands for dry run, real run, and HTML comparison.

Current state:

- These files are generated from existing local artifacts.
- No model call is made by this checker.
- The pre-read dry run has written prompts under:
  `/Users/clarkbenham/side_projects/make_structured_summaries/data/preread_chunk_notes/master-of-the-senate-master-of-the-senate`
