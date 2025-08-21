# src/prompt_mode/core.py
"""
Core orchestration for prompt-mode-min.

Implements:
- PromptModeV1: draft -> critique -> single revision
- PromptModeV2: plan -> (draft -> critique -> revision) * N with early stop

Design goals:
- Small, readable, and testable with LocalMock.
- Honest budgeting using rough token estimates (no tokenizer deps).
- Persist intermediate PassRecords; return a RunResult.

Notes:
- Prompts are loaded from ./prompts/*. Keep them short and auditable.
- "Timeouts" are cooperative; adapters should keep calls fast in tests.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .llm import LLM, Message
from .schemas import (
    PassRecord,
    RunResult,
    V1Config,
    V2Config,
)
from . import utils


# -------------------------
# Prompt loading
# -------------------------

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _read_prompt_file(name: str, default: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.exists():
        return default.strip()
    return path.read_text(encoding="utf-8").strip()


_SYSTEM_V1 = _read_prompt_file(
    "system_v1.txt",
    """
You are an LLM that performs a single self-critique and revision.

Process:
1) Produce a concise DRAFT answer to the user's request.
2) Critique your draft against the request and constraints (clarity, coverage, constraints).
3) Produce a REVISED answer that applies the critique. Keep it short and structured.

Output only the revised answer (no meta commentary).
""",
)

_SYSTEM_V2 = _read_prompt_file(
    "system_v2.txt",
    """
You are an LLM that plans, iterates, and improves an answer in small passes.

Process:
- PLAN: outline 2–4 subgoals needed to answer well.
- For each pass: propose a short DRAFT, request/consider CRITIQUE, then REVISE.
- Keep answers concise and structured. Avoid scope creep. Respect constraints.

You will be given a separate CRITIC to review drafts during passes.
""",
)

_CRITIC_GUIDELINES = _read_prompt_file(
    "critic_guidelines.txt",
    """
You are a CRITIC. Evaluate ONLY the candidate answer against the user's request.

Score with bullets in 3 dimensions (0.00–1.00):
- Coverage — does it answer the full ask?
- Clarity — is it concise and readable?
- Constraints — does it obey explicit constraints?

Then give 2–3 concrete improvement suggestions.
Finish with line: **Overall**: <score>
""",
)


# -------------------------
# Helpers
# -------------------------

def _parse_overall_score(text: str) -> Optional[float]:
    """
    Extract '**Overall**: 0.87' style score from critic output.
    Returns None if not found.
    """
    m = re.search(r"\*\*Overall\*\*:\s*([01](?:\.\d+)?)", text)
    if not m:
        # tolerate 'Overall:' without bold
        m = re.search(r"\bOverall:\s*([01](?:\.\d+)?)", text, flags=re.I)
    if not m:
        return None
    try:
        val = float(m.group(1))
        if 0.0 <= val <= 1.0:
            return val
    except Exception:
        pass
    return None


def _messages_with_budget(system_prompt: str, user_text: str, max_tokens: int) -> List[Message]:
    msgs: List[Message] = [
        {"role": "system", "content": utils.sanitize_text(system_prompt)},
        {"role": "user", "content": utils.sanitize_text(user_text)},
    ]
    return utils.truncate_messages(msgs, max_tokens=max_tokens, keep_system=True)


def _critic_messages(user_text: str, candidate_text: str) -> List[Message]:
    sys_prompt = f"{_CRITIC_GUIDELINES}\n\nYou will receive the user's request and a CANDIDATE answer."
    msgs: List[Message] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"USER REQUEST:\n{utils.sanitize_text(user_text)}\n\nCANDIDATE:\n{utils.sanitize_text(candidate_text)}"},
    ]
    # Budget generously here; orchestration will clamp via adapter max_tokens.
    return utils.truncate_messages(msgs, max_tokens=2000, keep_system=True)


def _revision_messages(system_prompt: str, user_text: str, draft: str, critique: str) -> List[Message]:
    sys_prompt = (
        f"{system_prompt}\n\n"
        "Revise the answer by APPLYING the critique below.\n"
        "Respond with ONLY the revised answer."
    )
    msgs: List[Message] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"USER REQUEST:\n{utils.sanitize_text(user_text)}"},
        {"role": "assistant", "content": f"DRAFT:\n{utils.sanitize_text(draft)}"},
        {"role": "user", "content": f"CRITIQUE:\n{utils.sanitize_text(critique)}\n\nPlease provide the REVISED answer now."},
    ]
    return utils.truncate_messages(msgs, max_tokens=2000, keep_system=True)


# -------------------------
# V1 Orchestrator
# -------------------------

@dataclass
class PromptModeV1:
    model: LLM
    config: V1Config = V1Config()

    def run(self, task_text: str) -> RunResult:
        started = utils._utc_now_iso() if hasattr(utils, "_utc_now_iso") else None  # tolerate direct run
        passes: List[PassRecord] = []
        token_total = 0
        stopped_reason = "complete"
        error_message = None

        try:
            # 1) DRAFT
            msgs = _messages_with_budget(_SYSTEM_V1, task_text, self.config.max_input_tokens)
            draft = self.model.generate(
                msgs,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                timeout_seconds=self.config.timeout_seconds,
            )
            token_total += utils.rough_messages_token_count(msgs) + utils.rough_token_count(draft)

            # 2) CRITIQUE
            cmsgs = _critic_messages(task_text, draft)
            critique = self.model.generate(
                cmsgs,
                temperature=0.0,  # stable critic
                max_tokens=256,
                timeout_seconds=self.config.timeout_seconds,
            )
            token_total += utils.rough_messages_token_count(cmsgs) + utils.rough_token_count(critique)

            # 3) REVISION
            rmsgs = _revision_messages(_SYSTEM_V1, task_text, draft, critique)
            revision = self.model.generate(
                rmsgs,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
                timeout_seconds=self.config.timeout_seconds,
            )
            token_total += utils.rough_messages_token_count(rmsgs) + utils.rough_token_count(revision)

            diff = utils.diff_text(draft, revision)
            passes.append(
                PassRecord(
                    step=1,
                    phase="revision",
                    draft=draft,
                    critique=critique,
                    revision=revision,
                    diff=diff,
                    token_estimate=token_total,
                    meta={"mode": "v1"},
                )
            )

            final = revision.strip() or draft.strip()
        except Exception as e:
            stopped_reason = "error"
            error_message = str(e)
            final = (passes[-1].revision if passes else "").strip() or "ERROR: " + error_message

        finished = utils._utc_now_iso() if hasattr(utils, "_utc_now_iso") else None

        return RunResult(
            mode="v1",
            final_output=final,
            passes=passes,
            token_count=token_total,
            stopped_reason=stopped_reason,  # type: ignore[arg-type]
            error_message=error_message,
            started_at=started or "",
            finished_at=finished or "",
            config_snapshot=self.config.model_dump(),
        )


# -------------------------
# V2 Orchestrator
# -------------------------

@dataclass
class PromptModeV2:
    model: LLM
    max_passes: int = 3
    config: V2Config = V2Config()

    def run(self, task_text: str) -> RunResult:
        started = utils._utc_now_iso() if hasattr(utils, "_utc_now_iso") else None
        passes: List[PassRecord] = []
        token_total = 0
        stopped_reason = "complete"
        error_message = None

        # plan step (lightweight, inlined to keep code small)
        plan_msgs = _messages_with_budget(_SYSTEM_V2, f"Plan the answer as 2–4 bullet subgoals.\n\nTask:\n{task_text}", self.config.max_input_tokens)
        try:
            plan = self.model.generate(
                plan_msgs,
                temperature=0.1,
                max_tokens=200,
                timeout_seconds=self.config.timeout_seconds,
            )
            token_total += utils.rough_messages_token_count(plan_msgs) + utils.rough_token_count(plan)
        except Exception as e:
            # If planning fails, proceed without plan (still honest)
            plan = "• Provide concise answer\n• Cover constraints\n• Include rationale\n"
            error_message = f"plan_error: {e}"

        try:
            for step in range(1, max(1, min(self.max_passes, self.config.max_passes)) + 1):
                t0 = time.time()

                # DRAFT for this pass (keep it small and iterative)
                draft_msgs: List[Message] = [
                    {"role": "system", "content": _SYSTEM_V2},
                    {"role": "user", "content": f"USER REQUEST:\n{utils.sanitize_text(task_text)}\n\nPLAN:\n{utils.sanitize_text(plan)}\n\nProvide a concise draft for pass {step}."},
                ]
                draft_msgs = utils.truncate_messages(draft_msgs, max_tokens=self.config.max_input_tokens, keep_system=True)

                draft = self.model.generate(
                    draft_msgs,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    timeout_seconds=self.config.timeout_seconds,
                )
                token_total += utils.rough_messages_token_count(draft_msgs) + utils.rough_token_count(draft)

                # CRITIQUE
                cmsgs = _critic_messages(task_text, draft)
                critique = self.model.generate(
                    cmsgs,
                    temperature=0.0,
                    max_tokens=256,
                    timeout_seconds=self.config.timeout_seconds,
                )
                token_total += utils.rough_messages_token_count(cmsgs) + utils.rough_token_count(critique)

                # REVISION
                rmsgs = _revision_messages(_SYSTEM_V2, task_text, draft, critique)
                revision = self.model.generate(
                    rmsgs,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    timeout_seconds=self.config.timeout_seconds,
                )
                token_total += utils.rough_messages_token_count(rmsgs) + utils.rough_token_count(revision)

                diff = utils.diff_text(draft, revision)
                elapsed_ms = int((time.time() - t0) * 1000)

                passes.append(
                    PassRecord(
                        step=step,
                        phase="revision",
                        plan=plan,
                        draft=draft,
                        critique=critique,
                        revision=revision,
                        diff=diff,
                        token_estimate=token_total,
                        elapsed_ms=elapsed_ms,
                        meta={"mode": "v2"},
                    )
                )

                # Early stop if critic says we're good enough
                score = _parse_overall_score(critique) or 0.0
                if self.config.early_stop_score is not None and score >= float(self.config.early_stop_score):
                    stopped_reason = "early_stop"
                    break

                # Budget guard (simple heuristic)
                if token_total >= (self.config.max_input_tokens + self.config.max_output_tokens) * 4:
                    stopped_reason = "token_budget"
                    break

                # Max passes guard is covered by loop bounds

            final = (passes[-1].revision if passes else "").strip()
            if not final:
                # Fallback: try drafting once if something went off
                fm = _messages_with_budget(_SYSTEM_V2, task_text, self.config.max_input_tokens)
                final = self.model.generate(
                    fm,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                    timeout_seconds=self.config.timeout_seconds,
                ).strip()
                token_total += utils.rough_messages_token_count(fm) + utils.rough_token_count(final)

        except Exception as e:
            stopped_reason = "error"
            error_message = str(e)
            final = (passes[-1].revision if passes else "").strip() or "ERROR: " + error_message

        finished = utils._utc_now_iso() if hasattr(utils, "_utc_now_iso") else None

        return RunResult(
            mode="v2",
            final_output=final,
            passes=passes,
            token_count=token_total,
            stopped_reason=stopped_reason,  # type: ignore[arg-type]
            error_message=error_message,
            started_at=started or "",
            finished_at=finished or "",
            config_snapshot=self.config.model_dump(),
        )
