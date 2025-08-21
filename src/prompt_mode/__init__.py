# src/prompt_mode/__init__.py
"""
prompt-mode-min

Minimal demo of Prompt Mode v1 (single self-critique) and
Prompt Mode v2 (plan + multi-pass critique).

Exports:
    PromptModeV1, PromptModeV2 for orchestration
    V1Config, V2Config for configuration
    RunResult, PassRecord for artifacts
"""

__version__ = "0.1.0"

from .core import PromptModeV1, PromptModeV2
from .schemas import V1Config, V2Config, RunResult, PassRecord

__all__ = [
    "PromptModeV1",
    "PromptModeV2",
    "V1Config",
    "V2Config",
    "RunResult",
    "PassRecord",
]
