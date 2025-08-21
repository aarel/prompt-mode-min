# src/prompt_mode/schemas.py
"""
Pydantic models for prompt-mode-min.

These models are:
- JSONL-friendly (stable field names, no complex unions needed).
- Strict enough to catch mistakes (e.g., negative caps).
- Small and readable in one sitting.

Used by:
- core.py (to build PassRecord + RunResult artifacts)
- cli.py (to serialize transcripts)
- evals/run_eval.py (to store tiny rubric scores)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


# -------------------------
# Common utilities
# -------------------------


def _utc_now_iso() -> str:
    """Return current UTC timestamp as ISO8601 string with 'Z'."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


# -------------------------
# Configs
# -------------------------


class BaseConfig(BaseModel):
    """
    Base parameters shared by V1/V2 runs.
    Keep it small—the point is orchestration clarity, not a framework.
    """

    model_config = ConfigDict(extra="forbid")

    max_input_tokens: int = Field(
        default=2000, ge=1, description="Rough budget for prompt context."
    )
    max_output_tokens: int = Field(
        default=512, ge=1, description="Rough cap per generation."
    )
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Sampling temperature."
    )
    timeout_seconds: int = Field(
        default=30, ge=1, description="Fail fast. Keep demos snappy."
    )


class V1Config(BaseConfig):
    """Single self-critique + revision."""
    # Nothing extra for V1 (deliberately simple).


class V2Config(BaseConfig):
    """Planner + multi-pass critique/revision."""
    max_passes: int = Field(
        default=3, ge=1, description="Upper bound on improvement iterations."
    )
    early_stop_score: Optional[float] = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="If critic/rubric claims we're 'good enough', stop early.",
    )


# -------------------------
# Artifacts per pass
# -------------------------


class PassRecord(BaseModel):
    """
    A single improvement pass. Stored line-by-line in JSONL for auditability.

    Fields:
      - step: 1-based index of the pass (or 0 for initial draft in V1)
      - plan: optional outline for V2 (None for V1)
      - draft: candidate text *before* critique
      - critique: feedback text from critic prompt (or heuristic)
      - revision: candidate text *after* applying critique
      - diff: unified diff between draft and revision (utils.diff_text)
      - token_estimate: coarse count for budgeting
      - elapsed_ms: wall time for this pass (set by core; optional)
      - meta: lightweight metadata bag
    """

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    phase: Literal["draft", "critique", "revision", "finalize"] = "revision"

    plan: Optional[str] = None
    draft: str
    critique: Optional[str] = None
    revision: str
    diff: str = ""

    token_estimate: int = Field(ge=0, default=0)
    elapsed_ms: Optional[int] = Field(default=None, ge=0)

    created_at: str = Field(default_factory=_utc_now_iso)
    meta: Dict[str, str] = Field(default_factory=dict)

    @field_validator("draft", "revision")
    @classmethod
    def _strip_spaces(cls, v: str) -> str:
        return v.strip()


# -------------------------
# Final result per run
# -------------------------


class RunResult(BaseModel):
    """
    Final bundle returned by PromptModeV1/PromptModeV2.run().

    Minimal but sufficient for the CLI + evals:
      - final_output: the last accepted text
      - passes: chronological PassRecords
      - token_count: rough total (inputs + outputs)
      - stopped_reason: 'max_passes', 'early_stop', 'error', etc.
      - config_snapshot: echo of the config used (V1Config/V2Config as dict)
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["v1", "v2"]
    final_output: str
    passes: List[PassRecord] = Field(default_factory=list)
    token_count: int = Field(ge=0, default=0)

    stopped_reason: Literal[
        "complete", "early_stop", "max_passes", "token_budget", "timeout", "error"
    ] = "complete"

    error_message: Optional[str] = None

    started_at: str = Field(default_factory=_utc_now_iso)
    finished_at: str = Field(default_factory=_utc_now_iso)

    config_snapshot: Dict[str, object] = Field(default_factory=dict)
    meta: Dict[str, str] = Field(default_factory=dict)

    @field_validator("final_output")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("final_output must not be empty")
        return v


# -------------------------
# Tiny eval score
# -------------------------


class EvalScore(BaseModel):
    """
    Minimal scoring record for small evals.
    This is intentionally vague; the repo is about orchestration, not SOTA eval.

    Example breakdown keys:
      - coverage (0..1)
      - clarity (0..1)
      - constraints (0..1)
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    mode: Literal["v1", "v2"]
    score_total: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    notes: Optional[str] = None
    created_at: str = Field(default_factory=_utc_now_iso)

    @field_validator("breakdown")
    @classmethod
    def _valid_breakdown(cls, v: Dict[str, float]) -> Dict[str, float]:
        for k, val in v.items():
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"breakdown[{k}] must be in [0.0, 1.0]")
        return v
