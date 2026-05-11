from pathlib import Path

from modules import profile_context


def test_load_profile_context_reads_configured_text_file(tmp_path, monkeypatch):
    context_file = tmp_path / "profile.txt"
    context_file.write_text("Applicant Name\nMarketing strategy", encoding="utf-8")
    monkeypatch.setenv("AUTO_APPLIER_PROFILE_CONTEXT_PATH", str(context_file))
    profile_context.load_profile_context.cache_clear()

    loaded = profile_context.load_profile_context()

    assert f"Profile context source: {Path(context_file)}" in loaded
    assert "Marketing strategy" in loaded
