# tests/test_cli_smoke.py
import os
import sys
from pathlib import Path
import json

import pytest # pyright: ignore[reportMissingImports]

# Ensure the parent directory is in sys.path so 'prompt_mode' can be imported
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# We call the CLI's main() directly to avoid spawning a new Python process.
from prompt_mode.cli import main as cli_main # pyright: ignore[reportMissingImports]


@pytest.mark.timeout(10)
def test_cli_v1_mock_creates_transcript_and_prints_output(tmp_path, capsys):
    # Arrange
    repo_root = Path(__file__).resolve().parents[1]
    task_file = repo_root / "examples" / "tasks" / "email_tone_fix.md"
    assert task_file.exists(), "Expected examples/tasks/email_tone_fix.md to exist"

    out_path = tmp_path / "v1_run.jsonl"

    # Force mock usage even if someone forgets --mock in CI
    os.environ["PM_FORCE_MOCK"] = "1"

    # Simulate CLI argv
    argv = [
        "python",
        "--mode",
        "v1",
        "--task",
        str(task_file),
        "--mock",
        "--save",
        str(out_path),
    ]

    # Act
    # argparse inside cli_main() reads sys.argv, so patch it.
    old_argv = sys.argv[:]
    sys.argv = ["prompt_mode"] + argv[1:]
    try:
        cli_main()
    finally:
        sys.argv = old_argv
        os.environ.pop("PM_FORCE_MOCK", None)

    # Assert stdout contained expected sections
    captured = capsys.readouterr().out
    assert "=== FINAL OUTPUT ===" in captured
    assert "Passes:" in captured
    assert "Token estimate:" in captured

    # Assert transcript was written and is valid JSONL
    assert out_path.exists(), "Transcript file was not created"
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1, "Transcript should have at least one pass"
    # Ensure each line is valid JSON with required keys
    for line in lines:
        obj = json.loads(line)
        assert "draft" in obj and "revision" in obj, "Transcript line missing expected fields"


@pytest.mark.timeout(10)
def test_cli_v2_mock_runs_multiple_passes(tmp_path, capsys):
    repo_root = Path(__file__).resolve().parents[1]
    task_file = repo_root / "examples" / "tasks" / "sql_query_review.md"
    assert task_file.exists(), "Expected examples/tasks/sql_query_review.md to exist"

    out_path = tmp_path / "v2_run.jsonl"
    os.environ["PM_FORCE_MOCK"] = "1"

    argv = [
        "python",
        "--mode",
        "v2",
        "--task",
        str(task_file),
        "--mock",
        "--passes",
        "2",
        "--save",
        str(out_path),
    ]

    old_argv = sys.argv[:]
    sys.argv = ["prompt_mode"] + argv[1:]
    try:
        cli_main()
    finally:
        sys.argv = old_argv
        os.environ.pop("PM_FORCE_MOCK", None)

    captured = capsys.readouterr().out
    assert "=== FINAL OUTPUT ===" in captured

    # V2 should produce >=1 pass; with --passes 2 it's typically 1–2 depending on early stop
    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    first = json.loads(lines[0])
    # V2 pass records should include a plan and meta.mode == "v2"
    assert first.get("plan"), "V2 pass should include a plan"
    assert first.get("meta", {}).get("mode") == "v2"
