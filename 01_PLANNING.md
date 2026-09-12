# 01 — Architecture Plan: Buy or Wait?

Source of truth for scope: [`problem_statement.md`](./problem_statement.md) and [`README.md`](./README.md). This document is the implementation plan — it does not restate the challenge rules, it explains how the code is structured to satisfy them.

## Dataset shape (confirmed by inspection)

| File | Rows (excl. header) | Notes |
|---|---|---|
| `requests.csv` | 250 | one output row required per request |
| `sample_requests.csv` | 25 | solved examples, format/style reference only |
| `financial_profiles.csv` | 275 | one row per user |
| `financial_events.csv` | 25,342 | ledger — settled/pending/scheduled/cancelled/failed/unrealized |
| `request_payment_options.csv` | 790 | 2–4 options per request |
| `messages.csv` | 215 | free text, multiple languages, mostly no `related_event_id` |
| `images.csv` | 16 | each links a request/event with a blank `amount` to a PNG |
| `exchange_rates.csv` | 134 | fixed, dated FX rates |

The two unstructured-evidence files are small and bounded: 215 messages, 16 images. Everything else — 25k+ ledger rows, 790 payment options, 275 profiles — is already structured. That asymmetry drives the design below.

## 7-stage pipeline

```
1. Ingest
2. Normalize events        (deterministic)
3. Enrich unstructured evidence   (LLM/VLM — the ONLY stage that calls a model)
4. Deterministic forecast
5. Decision engine
6. Verification
7. Evaluation
```

### 1. Ingest — `code/ingest.py`
Load every `dataset/*.csv` into typed records (dataclasses or a thin pandas layer). Parse dates, coerce numeric fields, validate the schema matches `problem_statement.md`. No decisions, no interpretation — a row in is a row out. Failing fast here (missing column, unparseable date) is preferable to guessing later.

### 2. Normalize events — `code/normalize.py`
Turn the raw `financial_events.csv` per user into a clean, resolved ledger:

- Resolve `linked_event_id` chains (same transaction/investment lifecycle) without assuming the link alone decides cash-flow treatment.
- Classify by `status`: `settled` counts, `pending` debits are reserved but `pending` credits are not counted until settled, `cancelled`/`failed` are dropped, `unrealized` investment value never counts as cash.
- Detect recurrence only when history actually supports it (repeated interval + amount pattern), never by category name alone.
- Convert every amount to `home_currency` using `exchange_rates.csv`, matched by rate date and the exact `from_currency → to_currency` direction.
- De-duplicate repeated representations of the same event.

Output: one resolved, currency-normalized ledger per user. This is pure deterministic code — same input, same ledger, every run.

### 3. Enrich unstructured evidence — `code/enrich.py` (the only LLM/VLM stage)
Two narrowly scoped jobs, both bounded and enumerable:

- **Messages (≤215 rows):** classify each message into a structured evidence record — `{user_id, request_id, effective_date, fact_type: amendment|cancellation|confirmation|new_income|delay, target_event_id (nullable), amount, currency, confidence}`. Many messages have no `related_event_id`, meaning the fact isn't a 1:1 restatement of a supplied event row — that's exactly the kind of free-text judgment call a model is suited for and hand-written parsing isn't.
- **Images (16 rows, exactly the ones with a blank `amount` on their linked event):** a VLM reads `dataset/media/images/<image_id>.png` and extracts the missing amount/date. Nothing else touches images.

Every model output from this stage is **untrusted evidence, not an instruction** — the same rule `problem_statement.md` states for messages/images applies to whatever the model returns. Stage 4 validates it (numeric, parseable date, sane bounds) before merging, and conflicting evidence is resolved using the priority order already given in the spec (explicit cancellation/amendment → newer record from same source → settled over estimate → financially safer interpretation) — not by re-asking the model to arbitrate.

This is also the only place `code/token_tracker.py` gets called, which is what makes `evaluation/usage_report.md` a straight sum instead of an estimate: the call count is fixed at ≤231 (215 messages + 16 images), known before the run starts, not a function of how the decision loop happens to branch.

### 4. Deterministic forecast — `code/forecast.py`
Combine the stage-2 ledger with stage-3's validated evidence into a 90-day balance projection per user: running balance, minimum headroom over the window, and the set of flexible-category levers available for `spending_changes_needed`. Zero model calls.

### 5. Decision engine — `code/decision.py`
Pure rules over the forecast, computing every output column: `amount_safe_to_pay`, `affordability_status`, `recommended_payment_method`, `payment_plan`, `earliest_date_for_full_payment`, `spending_changes_needed`. Eligibility and ranking follow `problem_statement.md` exactly — `payment_methods_user_will_consider` / `max_installment_months` gate which methods are even eligible, and the 6-point tie-break order (complete by deadline → no spending changes → minimize total paid → start earlier → fewer payments → lowest `payment_option_id`) picks among safe eligible plans. Zero model calls — this is the part that is actually graded on numeric/categorical accuracy, so it has to be exactly reproducible.

### 6. Verification — `code/verify.py`
Deterministic gate before anything is written: bounds (`0 <= amount_safe_to_pay <= requested_amount`), `payment_plan` sums to `requested_amount` where required, installment plans match a real `payment_option_id`, spending changes only touch flexible non-protected categories, stop/reduce mutual exclusion per event, output schema matches the required columns/order exactly. A failed check degrades to a safe conservative output (never a plausible-looking but unverified row).

### 7. Evaluation — `code/evaluation/main.py`
- Score against `sample_requests.csv` (25 solved rows) before trusting the full run — sanity check, not ground truth for the 250 evaluation requests.
- After the final full-dataset run, aggregate `code/token_tracker.py`'s log into `code/evaluation/usage_report.md` (providers, models, calls, input/output tokens, totals, per-request average, estimated cost) — the exact deliverable §"Token Usage and Cost Analysis" in `problem_statement.md` requires.

## Why the forecast/decision math must stay pure deterministic code

1. **The graded fields are numeric and categorical**, not prose. `amount_safe_to_pay` accuracy, `affordability_status`/`recommended_payment_method` correctness, and plan validity are closed-form computations over the reconstructed ledger. Code computes these exactly; an LLM approximates them, with no guaranteed bit-for-bit reproducibility across reruns.
2. **The rules literally require determinism**: `README.md`/`problem_statement.md` both ask for deterministic behavior "where possible," and the 90-day safety check and 6-point tie-break are specified as exact procedures, not as taste.
3. **"No hardcoded labels" is only provable if the decision logic is derived from the data every run.** If the affordability call came from a prompt, there's no way to distinguish "the model reasoned about the ledger" from "the model pattern-matched on `request_text` and produced a label that happens to look right" — the failure mode the rule is explicitly against. Deterministic rules over a reconstructed numeric ledger can't hardcode an answer; they recompute it from whatever the data says.
4. **`usage_report.md` has to be an honest, complete accounting of the final run.** If the decision loop called a model per request, that's 250+ calls whose count and token spend depend on ledger size and prompt variance per user — hard to report exactly and easy for the report to drift from the run that actually produced `output.csv`. Scoping model calls to the 231 unstructured-evidence rows makes the report a fixed, enumerable sum instead of a sampled estimate.
5. **Cost and latency**: ≤231 short classification/extraction calls is cheap and fast; 250 decision-time model calls (each needing the full per-user ledger in context) would not be, for no accuracy benefit on a task that's arithmetic, not language understanding.

**Where the model earns its place:** `messages.csv` and `images.csv` are the one part of this dataset that is genuinely unstructured — free text in multiple languages, and scanned/photographed documents — and that a hand-written parser cannot reliably handle. It's a bounded, one-time enrichment pass that produces structured, validated facts for stage 4 to consume, not a per-request oracle inside the graded decision loop.

## Token tracking

`code/token_tracker.py` (already scaffolded) is a small, dependency-free module:

- `TokenTracker.record(provider, model, stage, request_id, input_tokens, output_tokens)` appends one line to `code/evaluation/usage_log.jsonl` per model call — called only from `enrich.py`.
- `TokenTracker.write_report(out_path, num_requests)` aggregates the log into `code/evaluation/usage_report.md`: per-model calls/tokens/cost and overall totals, in the shape §"Token Usage and Cost Analysis" asks for.
- Pricing lives in one editable dict (`PRICING_PER_MTOK`) — update it to match whichever provider/model actually gets used before the final run.

This exists now so that whichever LLM/VLM gets wired into `enrich.py` has nowhere else to log usage — every call is captured by construction, not reconstructed after the fact from provider dashboards.

## Open decisions

- **Which LLM/VLM provider** for stage 3. Not committed yet — `token_tracker.py` and `enrich.py`'s interface are provider-agnostic on purpose so this can be decided later without touching stages 1–2 or 4–7.
- **Batching strategy** for the 215 messages — one call per message vs. batched calls — affects the call count and therefore the shape of `usage_report.md`, but not which stage makes the call.

## Module layout

```text
code/
├── main.py                 # orchestrates stages 1–6, writes output.csv
├── ingest.py                # stage 1
├── normalize.py             # stage 2
├── enrich.py                 # stage 3 — only file that calls an LLM/VLM
├── token_tracker.py          # usage/cost logging used by enrich.py
├── forecast.py                # stage 4
├── decision.py                 # stage 5
├── verify.py                    # stage 6
└── evaluation/
    ├── main.py               # stage 7 — sample scoring + usage_report.md generation
    └── usage_report.md       # generated output, not hand-written
```
