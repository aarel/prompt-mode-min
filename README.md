# prompt-mode-min

Minimal, readable demos of Prompt Mode v1 (single self-critique) and Prompt Mode v2 (plan + multi-pass critique) for LLM prompting.

**Why this exists**  
To show prompt-orchestration mechanics without faking a production platform.  
All code is dependency-light, testable offline, and readable in one sitting.

---

## Features
- **Prompt Mode V1** — draft → critique → single revision.
- **Prompt Mode V2** — plan → multiple critique/revision passes.
- Adapter pattern for model backends:
  - `LocalMock` for deterministic, network-free testing.
  - Optional OpenAI adapter for local experiments.
- Saved transcripts (JSONL) of every run for auditability.
- Tiny rubric-based eval script to compare V1 vs V2 trends.

---

## Quickstart

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run V1 with mock
python -m prompt_mode --mode v1 \
  --task examples/tasks/email_tone_fix.md \
  --mock \
  --save examples/transcripts/v1_email.jsonl

# Run V2 with mock
python -m prompt_mode --mode v2 \
  --task examples/tasks/sql_query_review.md \
  --mock \
  --passes 3 \
  --save examples/transcripts/v2_sql.jsonl
