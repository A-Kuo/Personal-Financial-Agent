# Buy or Wait?

This is my fork of the Hackerrank Orchestrate - Create a personalized financial spending agent.

Deterministic 90-day affordability forecaster with a narrowly-scoped LLM/VLM enrichment
stage. See [`../01_PLANNING.md`](../01_PLANNING.md) for the full architecture rationale
(why the forecast/decision math stays pure deterministic code, and why only
`enrich.py` ever calls a model).

## Setup

```bash
pip install -r ../requirements.txt
```

Create a `.env` file in the repository root (same directory as `AGENTS.md`) with:

```
MISTRAL_API_KEY=your-key-here
```

The key must be scoped for the **Ministral 3** family (`ministral-3b-latest`) and
Mistral OCR (`mistral-ocr-latest`). If a different tier's key returns a
`429 Rate limit exceeded` on `TEXT_MODEL` in `enrich.py`, run:

```bash
python -c "
from mistralai.client import Mistral
import os
from dotenv import load_dotenv
load_dotenv()
c = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
for m in c.models.list().data: print(m.id)
"
```

and update `TEXT_MODEL`/pricing constants in `enrich.py` to a model your key actually
has a nonzero `req/minute` allowance for (check the `429` response's
`x-ratelimit-limit-req-minute` header — `0` means the model is simply not on that
key's plan, not a transient throttle worth retrying).

## Run

From the repository root:

```bash
python code/main.py --force-refresh
```

This makes real Mistral calls, writes `output.csv` to the repository root, and writes
`code/evaluation/usage_report.md`. `--force-refresh` re-calls the model even for
already-cached extractions, so the usage report reflects real token/cost numbers
instead of $0 cache hits — use it for the final submission run. Omit it for a cheaper
rerun that reuses `code/cache/*.json`.

```bash
python code/main.py --skip-enrichment
```

Runs the deterministic stages only (ingest → normalize → forecast → decision →
verify), with `image_results`/`message_results` empty. Makes zero API calls — use this
to sanity-check the non-LLM 90% of the pipeline during development. **Never use this
flag to produce a submitted `output.csv`** — the 16 image-derived amounts and the
message-driven event corrections are real inputs to the decision, not optional
flourishes.

## Score against the solved samples

```bash
python -m code.evaluate
```

Runs the full pipeline against `dataset/sample_requests.csv`'s 25 solved rows and
prints per-field accuracy plus a table of mismatches. This is a sanity check on
decision *style*, not a proxy for the graded 250 — `problem_statement.md` is explicit
that the samples are examples of format, not labels for the evaluation requests.
Accepts `--skip-enrichment` too, for the same reason as above.

## Demo

<img width="943" height="642" alt="image" src="https://github.com/user-attachments/assets/8834ed11-70da-42f1-b6c8-33cce3f36a77" />

<img width="841" height="619" alt="image" src="https://github.com/user-attachments/assets/30b69178-8eb2-4425-9766-b81a4bcc986f" />



## Module layout

| File | Stage | Calls a model? |
|---|---|---|
| `ingest.py` | 1. Load and validate `dataset/*.csv` | No |
| `normalize.py` | 2. Resolve the ledger to clean cash flow (currency conversion, cancelled/failed exclusion, message-driven corrections) | No |
| `enrich.py` | 3. Classify messages (two-tier: ~30 deterministic phrase rules first, Mistral only for what doesn't match) and read receipt images via OCR | **Yes — the only file that does** |
| `forecast.py` | 4. Project a 90-day balance per user | No |
| `decision.py` | 5. Build every eligible, safe plan and rank by the spec's 6 tie-break criteria | No |
| `verify.py` | 6. Deterministic gate before anything is written | No |
| `evaluate.py` | 7. Score against the 25 solved samples | No |
| `token_tracker.py` | Logs every model call from `enrich.py`; `run_pipeline.py` renders it into `usage_report.md` | No |
| `run_pipeline.py` | Orchestrates 1–6, writes `output.csv` + `usage_report.md` | No |
| `main.py` | CLI entry point (`python code/main.py`) | No |

## Known limitations

- `amount_safe_to_pay` accuracy against the 25 solved samples is the softest metric
  (see `code/evaluation/usage_report.md`'s sibling accuracy run via `evaluate.py`).
  Several root causes were found and fixed this session (day-0-only balance instead
  of the 90-day minimum, plan-coupled instead of baseline capacity, unconverted
  foreign-currency income, a one-off temporary salary reduction being projected as
  the new permanent rate) — remaining gaps are most likely further instances of the
  same class of forecasting edge case rather than a single remaining defect.
- Two-tier message classification handles English and Indonesian, matching what's
  actually in `dataset/messages.csv`; the enrichment prompt also mentions Spanish as a
  Tier 2 fallback hint in case an unseen message doesn't match a Tier 1 rule.
