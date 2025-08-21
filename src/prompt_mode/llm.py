# src/prompt_mode/llm.py
"""
LLM interface + adapters for prompt-mode-min.

- LLM: minimal interface with a single `generate(...)` method.
- LocalMock: deterministic, offline responses for tests/demos.
- OpenAIAdapter: optional; disabled in CI and when NO_NETWORK/PM_FORCE_MOCK is set.

Design goals:
- Keep this file tiny and dependency-light.
- Fail LOUDLY if a real network/model is attempted in CI.
- Provide enough deterministic behavior to prove orchestration logic.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Protocol, Optional


Message = Dict[str, str]  # {"role": "...", "content": "..."}


class LLM(Protocol):
    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: int = 30,
    ) -> str:
        """
        Return the assistant's text for the given chat messages.
        Must NOT mutate `messages`.
        """
        ...


# -------------------------
# Helpers
# -------------------------

_CRITIC_HINTS = (
    "critic",
    "critique",
    "rubric",
    "reviewer",
    "score",
)

_EMAIL_HINTS = ("email", "tone", "polite", "professional")
_SQL_HINTS = ("select", "join", "where", "group by", "sql", "query")
_BUG_HINTS = ("bug", "issue", "stack trace", "exception", "repro", "steps to reproduce")


def _is_critic_mode(messages: List[Message]) -> bool:
    text = " ".join(m.get("content", "") for m in messages if m.get("role") == "system").lower()
    return any(h in text for h in _CRITIC_HINTS)


def _last_user_text(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _hash_ratio(text: str, lo: float = 0.6, hi: float = 0.95) -> float:
    """
    Map text -> deterministic float in [lo, hi].
    Used to make the mock's scores look "varied" but reproducible.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big") / float(2**64 - 1)
    return lo + (hi - lo) * n


def _truncate_paragraphs(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars)].rstrip() + " …[truncated]"


# -------------------------
# LocalMock (deterministic, offline)
# -------------------------

@dataclass
class LocalMock(LLM):
    """
    Deterministic, domain-aware offline mock.

    Behaviors:
    - If system prompt looks like a CRITIC, returns a critique with 3 bullets + scores.
    - Else produces a revision/answer in one of a few templates:
        * Email/tone: rewrites with concise, professional tone.
        * SQL: flags naive patterns and suggests a corrected query + rationale.
        * Bug summary: extracts likely cause + steps.
    - Injects a stable token "[MOCK]" so golden tests can assert determinism.

    NOTE: This is intentionally simple. It's here to exercise orchestration code.
    """

    tag: str = "[MOCK]"

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: int = 30,
    ) -> str:
        # Simulate small, bounded latency to catch accidental timeouts
        time.sleep(min(0.02, timeout_seconds / 1000.0))

        # Determine mode
        critic_mode = _is_critic_mode(messages)
        user = _last_user_text(messages)
        user_lc = user.lower()

        if critic_mode:
            return self._make_critique(user, max_tokens)

        # Pick a domain template
        if any(h in user_lc for h in _EMAIL_HINTS):
            return self._make_email_revision(user, max_tokens)
        if any(h in user_lc for h in _SQL_HINTS):
            return self._make_sql_review(user, max_tokens)
        if any(h in user_lc for h in _BUG_HINTS):
            return self._make_bug_summary(user, max_tokens)

        # Fallback: a plain "improve for clarity" pass
        return self._make_generic_revision(user, max_tokens)

    # ----- Critic -----

    def _make_critique(self, user_text: str, max_tokens: int) -> str:
        coverage = _hash_ratio("cov:" + user_text, 0.6, 0.95)
        clarity = _hash_ratio("cla:" + user_text, 0.6, 0.95)
        constraints = _hash_ratio("con:" + user_text, 0.55, 0.9)
        total = (coverage + clarity + constraints) / 3.0

        out = (
            f"{self.tag} Critique\n"
            f"- Coverage: {coverage:.2f} — Does it answer the full ask?\n"
            f"- Clarity: {clarity:.2f} — Is the structure concise and readable?\n"
            f"- Constraints: {constraints:.2f} — Adheres to explicit constraints?\n"
            f"**Overall**: {total:.2f}\n"
            f"Improvements:\n"
            f"1) Tighten wording; remove filler.\n"
            f"2) Ensure all constraints are addressed explicitly.\n"
            f"3) Add a short rationale before the final.\n"
        )
        return _truncate_paragraphs(out, max_tokens * 4)

    # ----- Email/Tone -----

    def _make_email_revision(self, user_text: str, max_tokens: int) -> str:
        out = (
            f"{self.tag} Revised Email (concise, professional):\n\n"
            "Subject: Follow-up on your request\n\n"
            "Hi [Name],\n\n"
            "Thanks for the update. Here’s the plan:\n"
            "• I’ll review the document and confirm next steps by EOD tomorrow.\n"
            "• If priorities changed, let me know and I’ll adjust.\n\n"
            "Best,\n"
            "[Your Name]\n"
        )
        return _truncate_paragraphs(out, max_tokens * 4)

    # ----- SQL Review -----

    def _make_sql_review(self, user_text: str, max_tokens: int) -> str:
        # Cheap heuristics for "risky" patterns
        flags = []
        if re.search(r"select\s+\*", user_text, flags=re.I):
            flags.append("Avoid SELECT *; project only required columns.")
        if re.search(r"\bjoin\b", user_text, flags=re.I) and "on" not in user_text.lower():
            flags.append("JOIN without ON clause risks a Cartesian product.")

        fix_query = (
            "SELECT u.id, u.email, COUNT(o.id) AS orders\n"
            "FROM users u\n"
            "LEFT JOIN orders o ON o.user_id = u.id\n"
            "WHERE u.created_at >= DATE '2024-01-01'\n"
            "GROUP BY u.id, u.email\n"
            "ORDER BY orders DESC;"
        )

        out = f"{self.tag} SQL Review\nFindings:\n"
        out += "\n".join(f"- {f}" for f in (flags or ["No obvious structural issues found."]))
        out += (
            "\n\nSuggested Query:\n```sql\n"
            f"{fix_query}\n"
            "```\nRationale:\n"
            "- Projects specific columns for readability/perf.\n"
            "- LEFT JOIN with explicit ON prevents unintended row explosion.\n"
            "- WHERE bound keeps scans reasonable; GROUP BY matches projections.\n"
        )
        return _truncate_paragraphs(out, max_tokens * 4)

    # ----- Bug Summary -----

    def _make_bug_summary(self, user_text: str, max_tokens: int) -> str:
        out = (
            f"{self.tag} Bug Report Summary\n"
            "Likely Cause:\n- Null or unexpected type in input when parsing response.\n\n"
            "Impact:\n- Request fails intermittently; users see 500.\n\n"
            "Repro Steps:\n"
            "1) Start the service locally.\n"
            "2) Send a request with a missing optional field.\n"
            "3) Observe stack trace in logs.\n\n"
            "Fix:\n- Add input validation and default handling before parsing.\n"
            "- Extend test to include missing/None field case.\n"
        )
        return _truncate_paragraphs(out, max_tokens * 4)

    # ----- Generic -----

    def _make_generic_revision(self, user_text: str, max_tokens: int) -> str:
        out = (
            f"{self.tag} Revised:\n"
            "- Leads with the answer in 1–2 lines.\n"
            "- Breaks supporting points into bullets.\n"
            "- Ends with next steps or a clear takeaway.\n\n"
            "Answer:\n"
            "1) Main point stated up front.\n"
            "2) Key details with minimal filler.\n"
            "3) Close with action or summary.\n"
        )
        return _truncate_paragraphs(out, max_tokens * 4)


# -------------------------
# OpenAIAdapter (optional, local-only)
# -------------------------

@dataclass
class OpenAIAdapter(LLM):
    """
    Thin wrapper around OpenAI Chat Completions.

    - Respects NO_NETWORK / PM_FORCE_MOCK by refusing to initialize.
    - Requires `openai` package (new-style client). If not installed, raises.
    - Keep defaults conservative; this is for local sanity checks only.
    """

    api_key: str
    model: str = "gpt-4o-mini"
    _client: Optional[object] = None  # lazy

    def __post_init__(self):
        # CI / offline guards
        if os.getenv("NO_NETWORK") == "1" or os.getenv("PM_FORCE_MOCK") == "1":
            raise RuntimeError("Network use disabled by NO_NETWORK/PM_FORCE_MOCK. Use LocalMock.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "The 'openai' package is required for OpenAIAdapter. "
                "Install it locally (not in CI) and try again."
            ) from e
        self._client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: int = 30,
    ) -> str:
        if self._client is None:
            self.__post_init__()

        # Safety: shallow copy to avoid mutation-by-reference
        msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

        # Timeout is best-effort; the SDK may not support per-call timeouts directly.
        # Keep it simple; this is for local experiments only.
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAIAdapter call failed: {e}") from e

        choice = getattr(resp, "choices", [None])[0]
        if not choice or not getattr(choice, "message", None):
            raise RuntimeError("OpenAIAdapter returned no choices/message.")
        text = getattr(choice.message, "content", "") or ""
        if not text.strip():
            raise RuntimeError("OpenAIAdapter returned empty content.")
        return text.strip()
