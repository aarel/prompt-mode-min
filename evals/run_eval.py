#!/usr/bin/env python3
"""
run_eval.py — lightweight evaluation harness for Prompt Mode

Usage:
    python run_eval.py --mode v1 --task examples/tasks/email_tone_fix.md
    python run_eval.py --mode v2 --task examples/tasks/sql_query_review.md
"""

import argparse
import json
from pathlib import Path
from src.promptmode.core import PromptModeV1, PromptModeV2
from src.promptmode.llm import LocalMock


def load_task(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_transcript(lines, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def run_eval(mode: str, task_path: Path, out_dir: Path):
    task_text = load_task(task_path)
    model = LocalMock()

    if mode == "v1":
        engine = PromptModeV1(model)
    elif mode == "v2":
        engine = PromptModeV2(model)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    transcript = list(engine.run(task_text))

    # name output based on mode + task stem
    out_file = out_dir / f"{mode}_{task_path.stem}.jsonl"
    save_transcript(transcript, out_file)

    print(f"[OK] Saved transcript to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Prompt Mode eval on a task")
    parser.add_argument("--mode", required=True, choices=["v1", "v2"], help="Prompt Mode version")
    parser.add_argument("--task", required=True, type=Path, help="Path to task .md file")
    parser.add_argument(
        "--out", default="examples/transcripts", type=Path, help="Directory for transcript output"
    )
    args = parser.parse_args()

    run_eval(args.mode, args.task, args.out)


if __name__ == "__main__":
    main()
