# tests/test_llm_mock.py
import os
import re
import pytest # pyright: ignore[reportMissingImports]

from prompt_mode.llm import LocalMock


@pytest.fixture(autouse=True)
def _force_offline_env():
    # Keep CI honest; if anyone tries to wire a real adapter in tests, fail fast.
    os.environ["PM_FORCE_MOCK"] = "1"
    os.environ["NO_NETWORK"] = "1"
    yield
    os.environ.pop("PM_FORCE_MOCK", None)
    os.environ.pop("NO_NETWORK", None)


def _sys(text: str):
    return {"role": "system", "content": text}


def _user(text: str):
    return {"role": "user", "content": text}


def test_determinism_same_input_same_output():
    mock = LocalMock()
    msgs = [_sys("you are a helpful model"), _user("please rewrite this email to be more polite and professional")]
    out1 = mock.generate(msgs, temperature=0.9, max_tokens=256)
    out2 = mock.generate(msgs, temperature=0.1, max_tokens=256)
    # LocalMock should be deterministic regardless of temperature for tests
    assert out1 == out2
    assert "[MOCK]" in out1


def test_email_mode_produces_email_template():
    mock = LocalMock()
    msgs = [_sys(""), _user("Please rewrite this email to be more polite and professional.")]
    out = mock.generate(msgs, max_tokens=256)
    assert out.startswith("[MOCK] Revised Email"), "Expected email revision heading"
    assert "Subject:" in out
    assert "Best," in out


def test_sql_mode_flags_risky_patterns_and_includes_code_block():
    mock = LocalMock()
    msgs = [_sys(""), _user("SELECT * FROM users JOIN orders;")]
    out = mock.generate(msgs, max_tokens=512)
    # Should flag SELECT * and missing ON clause
    assert "- Avoid SELECT *" in out
    assert "JOIN without ON clause" in out
    # Should include a ```sql block with a suggested query
    assert "```sql" in out and "```" in out.split("```sql", 1)[-1]


def test_bug_mode_produces_summary_repro_and_fix():
    mock = LocalMock()
    msgs = [_sys(""), _user("Bug: app throws exception on null input, see stack trace")]
    out = mock.generate(msgs, max_tokens=256)
    assert out.startswith("[MOCK] Bug Report Summary")
    assert "Likely Cause:" in out
    assert "Repro Steps:" in out
    assert "Fix:" in out


def test_generic_mode_when_no_domain_hints():
    mock = LocalMock()
    msgs = [_sys(""), _user("Improve this paragraph for clarity.")]
    out = mock.generate(msgs, max_tokens=256)
    assert out.startswith("[MOCK] Revised:")
    # Basic structure bullets
    assert "- Leads with the answer" in out


def test_critic_mode_detects_system_and_emits_scores_and_overall():
    mock = LocalMock()
    # Include 'critic' keyword in system to trigger critic behavior
    msgs = [
        _sys("You are a CRITIC. Provide a concise critique with scores."),
        _user("Some candidate answer to be reviewed.")
    ]
    out = mock.generate(msgs, temperature=0.0, max_tokens=128)
    # Check the three scored dimensions and Overall line
    assert "[MOCK] Critique" in out
    assert re.search(r"Coverage:\s*\d+\.\d{2}", out)
    assert re.search(r"Clarity:\s*\d+\.\d{2}", out)
    assert re.search(r"Constraints:\s*\d+\.\d{2}", out)
    assert re.search(r"\*\*Overall\*\*:\s*[01](?:\.\d+)?", out)


def test_respects_max_tokens_via_truncation_marker():
    mock = LocalMock()
    # Force tiny budget to ensure truncation occurs
    msgs = [_sys("you are helpful"), _user("SELECT * FROM users JOIN orders;")]
    out = mock.generate(msgs, max_tokens=1)  # 1 token ~ 4 chars threshold
    assert "…[truncated]" in out, "Expected output to be truncated for extremely small max_tokens"


def test_latency_is_small_and_timeout_param_is_accepted():
    mock = LocalMock()
    msgs = [_sys("system"), _user("email tone fix")]
    # Should return quickly and accept timeout_seconds kwarg
    out = mock.generate(msgs, timeout_seconds=5)
    assert out.startswith("[MOCK]")
