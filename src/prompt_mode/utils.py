# src/prompt_mode/utils.py
"""
Utilities for prompt-mode-min:
- Rough token estimates (dependency-free).
- Truncation for chat-style messages.
- Basic sanitization to avoid prompt breakage.
- Human-readable diffs for artifacts.

Deliberately simple and opinionated. This is glue, not a tokenizer.
"""

from __future__ import annotations

import difflib
import html
import math
import re
from typing import Dict, List, Sequence, Tuple


# -------------------------
# Token-ish estimation
# -------------------------

AVG_CHARS_PER_TOKEN = 4  # rough heuristic for English prose
_MIN_TOKEN = 1


def rough_token_count(text: str) -> int:
    """
    Extremely rough token estimate with no external deps.

    Heuristic:
      tokens ~= ceil(len(text) / AVG_CHARS_PER_TOKEN)
    We floor at 1 when text isn't empty.

    This deliberately *overestimates* on short text and underestimates a bit on long text.
    Good enough for pass caps and truncation decisions.

    >>> rough_token_count("hello world")
    3
    """
    if not text:
        return 0
    return max(_MIN_TOKEN, math.ceil(len(text) / AVG_CHARS_PER_TOKEN))


def rough_messages_token_count(messages: Sequence[Dict[str, str]]) -> int:
    """
    Sum rough token counts for a list of {"role": "...", "content": "..."} dicts.
    Roles are ignored for tokens; content only.
    """
    total = 0
    for m in messages:
        total += rough_token_count(str(m.get("content", "")))
    return total


# -------------------------
# Truncation
# -------------------------

def truncate_text_to_tokens(text: str, max_tokens: int) -> Tuple[str, int]:
    """
    Truncate a single string to approximately fit a token budget.

    Strategy:
      - Quick return if already within budget.
      - Hard cut by chars using the same heuristic (tokens ~ len/4).
      - Append an ellipsis marker when truncated.

    Returns (truncated_text, estimated_tokens).
    """
    if max_tokens <= 0:
        return "", 0

    current = rough_token_count(text)
    if current <= max_tokens:
        return text, current

    # Convert token budget to char budget
    char_budget = max(0, max_tokens * AVG_CHARS_PER_TOKEN)
    truncated = text[:char_budget].rstrip()
    if len(truncated) < len(text):
        truncated += " …[truncated]"
    return truncated, rough_token_count(truncated)


def truncate_messages(
    messages: List[Dict[str, str]],
    max_tokens: int,
    keep_system: bool = True,
) -> List[Dict[str, str]]:
    """
    Truncate a chat message list to roughly fit max_tokens.

    Policy:
      1) Optionally preserve the first system message in full (best practice).
      2) Keep newest messages; drop from the oldest user/assistant messages first.
      3) If still over budget, truncate the oldest remaining message content.
      4) As a last resort, truncate the final message.

    This is a pragmatic policy for small demos; not production-grade budgeting.

    Args:
      messages: list of {"role": "system|user|assistant", "content": "..."}
      max_tokens: rough budget
      keep_system: keep the *first* system message untouched when possible.

    Returns:
      New list of messages within budget (approx).
    """
    if max_tokens <= 0:
        return []

    msgs = [dict(m) for m in messages]  # shallow copy
    if not msgs:
        return msgs

    # Optionally pin the first system message
    pinned = []
    rest = msgs
    if keep_system and msgs[0].get("role") == "system":
        pinned = [msgs[0]]
        rest = msgs[1:]

    # Drop oldest non-pinned until within budget
    def current_tokens() -> int:
        return rough_messages_token_count(pinned + rest)

    while rest and current_tokens() > max_tokens:
        # Prefer dropping the oldest non-recent message (front of 'rest')
        # but never drop the *last* item (we need some prompt!)
        if len(rest) > 1:
            rest.pop(0)
        else:
            break

    # If still over, start truncating from the oldest remaining, then the last
    i = 0
    while current_tokens() > max_tokens and i < len(rest):
        content = str(rest[i].get("content", ""))
        # Aim to cut aggressively to converge
        remaining_budget = max_tokens - rough_messages_token_count(pinned + rest[:i] + rest[i + 1 :])
        remaining_budget = max(remaining_budget, 1)
        truncated, _ = truncate_text_to_tokens(content, remaining_budget)
        rest[i]["content"] = truncated
        i += 1

    # Final safety: if still over, truncate the newest (last) message
    if rest and current_tokens() > max_tokens:
        j = len(rest) - 1
        content = str(rest[j].get("content", ""))
        remaining_budget = max_tokens - rough_messages_token_count(pinned + rest[:j])
        remaining_budget = max(remaining_budget, 1)
        truncated, _ = truncate_text_to_tokens(content, remaining_budget)
        rest[j]["content"] = truncated

    return pinned + rest


# -------------------------
# Sanitization
# -------------------------

_NONPRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTILINE = re.compile(r"\n{3,}")


def sanitize_text(text: str) -> str:
    """
    Minimal sanitization to prevent common prompt/render breakage:
      - Strip control chars (except \n and \t).
      - Collapse excessive spaces/tabs and newlines.
      - Escape stray HTML that might confuse renderers (keeps content readable).
      - Guard against runaway code fences by normalizing backticks count.

    NOTE: This is intentionally conservative to avoid altering meaning.
    """
    if not text:
        return ""

    t = _NONPRINTABLE.sub("", text)
    t = _MULTISPACE.sub("  ", t)  # keep at most two spaces (readability)
    t = _MULTILINE.sub("\n\n", t)
    # Escape but preserve readability
    t = html.unescape(t)  # normalize any double-escaped entities the model might emit

    # Normalize wild code fences: keep at most 3 backticks sequences
    t = re.sub(r"`{4,}", "```", t)

    return t.strip()


# -------------------------
# Diffs for artifacts
# -------------------------

def diff_text(a: str, b: str, context: int = 2) -> str:
    """
    Produce a small unified diff between two strings for JSONL artifacts.

    Returns a single string. Empty string means "no changes".

    >>> diff_text("foo\\nbar\\n", "foo\\nbaz\\n")
    '--- a\\n+++ b\\n@@\\n-bar\\n+baz\\n'
    """
    if a == b:
        return ""

    a_lines = a.splitlines(keepends=False)
    b_lines = b.splitlines(keepends=False)
    udiff = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile="a",
        tofile="b",
        n=context,
        lineterm="",
    )
    out = "\n".join(udiff)
    return out


# -------------------------
# Small helpers
# -------------------------

def clamp(n: int, low: int, high: int) -> int:
    """Clamp integer n into [low, high]."""
    return max(low, min(high, n))


def coalesce(*values, default=None):
    """Return the first value that is not None, else default."""
    for v in values:
        if v is not None:
            return v
    return default
