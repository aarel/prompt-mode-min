# src/cli.py
"""
CLI entrypoint for prompt-mode-min.

Example:
    python -m prompt_mode --mode v1 --task examples/tasks/email_tone_fix.md --mock
"""

import argparse
import os
from pathlib import Path
import sys
import json

from prompt_mode.core import PromptModeV1, PromptModeV2
from prompt_mode.llm import LocalMock, OpenAIAdapter


def main():
    parser = argparse.ArgumentParser(description="Run Prompt Mode V1 or V2 on a task file.")
    parser.add_argument("--mode", choices=["v1", "v2"], required=True, help="Prompt Mode version to run.")
    parser.add_argument("--task", type=str, required=True, help="Path to task file (Markdown or text).")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes (V2 only).")
    parser.add_argument("--mock", action="store_true", help="Use LocalMock instead of real model.")
    parser.add_argument("--save", type=str, help="Path to save run transcript (JSONL).")
    args = parser.parse_args()

    task_path = Path(args.task)
    if not task_path.exists():
        sys.stderr.write(f"Task file not found: {task_path}\n")
        sys.exit(1)

    task_text = task_path.read_text(encoding="utf-8")

    # Model selection
    if args.mock or "PM_FORCE_MOCK" in os.environ:
        model = LocalMock()
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.stderr.write("No OPENAI_API_KEY found. Use --mock for local run.\n")
            sys.exit(1)
        model = OpenAIAdapter(api_key=api_key)

    # Orchestration
    if args.mode == "v1":
        runner = PromptModeV1(model)
        result = runner.run(task_text)
    elif args.mode == "v2":
        runner = PromptModeV2(model, max_passes=args.passes)
        result = runner.run(task_text)
    else:
        sys.stderr.write(f"Invalid mode: {args.mode}\n")
        sys.exit(1)

    # Output to stdout
    print("\n=== FINAL OUTPUT ===\n")
    print(result.final_output)
    print("\n=== SUMMARY ===")
    print(f"Passes: {len(result.passes)}")
    print(f"Token estimate: {result.token_count}")

    # Save transcript if requested
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            for pass_record in result.passes:
                f.write(json.dumps(pass_record.model_dump(), ensure_ascii=False) + "\n")
