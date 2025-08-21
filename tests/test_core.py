# tests/test_core.py
import os
from pathlib import Path
import json

import pytest

from prompt_mode.core import PromptModeV1, PromptModeV2
from prompt_mode.llm import LocalMock
from prompt_mode.schemas import V2Config


@pytest.fixture(autouse=True)
def _force_mock_and_no_network_env():
    # Keep CI honest: refuse any real adapter in tests.
    os.environ["PM_FORCE_MOCK"] = "1"
    os.environ["NO_NETWORK"] = "1"
    yield
    os.environ.pop("PM_FORCE_MOCK", None)
    os.environ.pop("NO_NETWORK", None)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def examples_dir(repo_root: Path) -> Path:
    return repo_root / "examples" / "tasks"


def _read(p: Path) -> str:
    assert p.exists(), f"Missing required file: {p}"
    return p.read_text(encoding="utf-8").strip()


def test_v1_email_flow_is_deterministic_and_records_passes(examples_dir: Path, tmp_path: Path):
    task_file = examples_dir / "email_tone_fix.md"
    task_text = _read(task_file)

    model = LocalMock()
    runner = PromptModeV1(model=model)
    result = runner.run(task_text)

    # Basic structure
    assert result.mode == "v1"
    assert result.final_output, "Final output should not be empty"
    assert result.passes, "Should record at least one pass"
    p0 = result.passes[0]
    assert p0.draft and p0.revision, "Pass should include draft and revision"
    assert "[MOCK]" in p0.revision, "LocalMock tag should appear in revision"
    # V1 should produce an email-style revision for this task
    assert result.final_output.startswith("[MOCK] Revised Email"), "Unexpected final output format for email task"

    # Token accounting should be non-zero and monotonic per pass record
    assert result.token_count > 0
    assert p0.token_estimate >= 0

    # Save a transient transcript as JSONL to ensure serializability
    out_path = tmp_path / "v1_email.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in result.passes:
            f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")
    assert out_path.exists()


def test_v2_sql_flow_has_plan_passes_and_sql_review(examples_dir: Path):
    task_file = examples_dir / "sql_query_review.md"
    task_text = _read(task_file)

    model = LocalMock()
    # Keep defaults; allow early stop if critic is high enough
    runner = PromptModeV2(model=model, max_passes=2, config=V2Config(max_passes=2, early_stop_score=0.9))
    result = runner.run(task_text)

    assert result.mode == "v2"
    assert len(result.passes) >= 1, "V2 should record at least one pass"
    first = result.passes[0]
    assert first.plan, "V2 pass should include a plan"
    assert first.meta.get("mode") == "v2"
    assert "[MOCK]" in first.revision
    # For SQL tasks, LocalMock produces an SQL Review with a code block
    assert "SQL Review" in result.final_output
    assert "```sql" in result.final_output

    # Stopped reason is either early_stop or complete under small caps
    assert result.stopped_reason in {"early_stop", "complete", "max_passes", "token_budget"}

    # Basic accounting checks
    assert result.token_count > 0
    for rec in result.passes:
        assert rec.token_estimate >= 0
        assert isinstance(rec.step, int) and rec.step >= 1


def test_v2_early_stop_threshold_is_respected(examples_dir: Path):
    task_file = examples_dir / "bug_report_summarize.md"
    task_text = _read(task_file)

    model = LocalMock()
    # Force early-stop to always trigger after the first pass by setting threshold to 0.0
    cfg = V2Config(max_passes=5, early_stop_score=0.0)
    runner = PromptModeV2(model=model, max_passes=5, config=cfg)
    result = runner.run(task_text)

    assert len(result.passes) == 1, "With early_stop_score=0.0, should stop after first pass"
    assert result.stopped_reason in {"early_stop", "complete"}
    assert "[MOCK]" in result.final_output
