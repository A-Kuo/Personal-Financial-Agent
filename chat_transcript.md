# Buy or Wait? — Curated Development Transcript

> A selection of the most actionable decision points from the engineering sessions that
> built this solution. The complete per-turn log is in `log.txt`. Timestamps are UTC.

**Deadline target:** 2026-09-13 18:00 IST (12:30 UTC).

---

## 1. Architecture decision — deterministic core, LLM scoped to unstructured input only

**Prompt:**
> "01_PLANNING.md — the real architecture: a 7-stage pipeline (ingest → normalize events →
> enrich unstructured text/images → deterministic forecast → decision engine →
> verification → evaluation), with an explicit argument for why the forecast/decision
> math must stay pure deterministic code while LLM/VLM use is narrowly scoped to the
> genuinely unstructured messages.csv and the 16 image-linked events."

**Outcome (why this matters):** Genuinely unstructured input is small and bounded — 215
messages, 16 images. Everything else (25k ledger events, 790 payment options, 275
profiles) is already structured. Keeping stages 4–6 as pure deterministic code is what
makes the "no hardcoded labels" and "usage_report corresponds to the final run" requirements
provable, and it bounds model cost to a fixed, enumerable set of calls.

---

## 2. Making the pipeline actually run — six correctness fixes at once

**Key fixes landed** (getting the codebase to a passing 250-request run):
- `config.py` was missing the cache paths `enrich.py` imports.
- `enrich.py` used `from mistralai import Mistral`, incompatible with the resolvable
  `mistralai>=2.10` (a namespace package) — corrected to `from mistralai.client import Mistral`.
- Image data URIs hardcoded `image/jpeg`; the dataset is always PNG — switched to mime-type detection.
- `decision.py` recommended `full_payment`/`wait` without checking
  `payment_methods_user_will_consider` — spec violation, added eligibility gating.
- `earliest_date_for_full_payment` used the installment plan's last date instead of raw
  lump-sum capacity — the spec states this field is independent of the chosen method.
- Machine fields (`payment_plan`, `spending_changes_needed`) used comma-thousands separators;
  the required format has none (found via a `verify.py` mismatch).
---

## 3. Two-tier message pipeline + token accounting + `safe_today` bug

**Key decisions:**
- Built a **two-tier message classifier**: Tier 1 is deterministic multilingual pattern
  rules covering the ~12 recurring fact categories (salary update / unpaid leave / bonus /
  refund / investment change / failed debit / rent increase / scam, etc.); **Tier 2** is a
  Mistral fallback (temperature 0, JSON-only, SHA-256 cached) **only** for messages that
  survive Tier 1 unclassified. Result: **212/215 messages resolved at zero model cost**; only
  3 hit the LLM.
- Diagnosed a misleading `429` (not quota exhaustion — that model had a 0 req/min
  entitlement on the key) and added retry-with-backoff plus fast-fail on permanent
  entitlements, and a 5 s OCR→chat cooldown.
- **`TokenTracker.reset()`** on every real run + a `--force-refresh` flag, so the `usage_report`
  reflects genuine non-cached token/cost — fixing a real bug where a leftover smoke-test log
  entry made the report contradict its own "this is the final run" claim.
- Fixed `safe_today` from "day-0 balance only" to the **forecast-wide minimum balance** —
  paying X today reduces every later day by X, so the true safe amount is the lowest point,
  not today's headroom.

---

## 4. Two more dataset-wide correctness bugs

**Prompt:** *(accuracy-regression chasing)*
> "Continue"

**Root causes found and fixed:**
- `normalize.py`'s cash-event constants were misspelled (`"debtpayment"` vs the real
  `"debt_payment"`; `investment_sale` missing) — this silently **dropped all 567 debt-payment,
  29 investment-purchase, and 5 investment-sale events dataset-wide**, making balances look
  healthier than reality for anyone with debt.
- `cancelled`/`failed` status events were never filtered out (a `"noncash"` vs `"non_cash"`
  typo), contradicting the spec's explicit "ignore failed or cancelled transactions" rule for
  the 90-day safety check — 43 such events existed.
- Income projection was **double-counting** (grouping by exact description split "Payroll
  credit" from the relabeled "Next confirmed salary" into two series) **and under-counting**
  (only historical events were used to detect recurrence, so confirmed future salaries were
  ignored in the forecast).
---

## 5. Systematic audit against the problem statement

**Prompt:**
> "audit systematically against problem statement"

**Findings (using `sample_requests.csv` as ground truth):**
- `amount_safe_to_pay` must be a **single uniform baseline** ("before optional spending
  changes", independent of the chosen plan). Proven by request_06 (pays 620.40 via a spending
  change but reports 603.30, the pre-change baseline). Unified to
  `min(safe_today, requested_amount)` across every branch.
- Installment completion was checked against the 90-day window instead of
  `desired_completion_date`; installment ranking was a `[financing_fee, total_payable]` proxy
  instead of the spec's literal 6 criteria.
- `reducibleorstoppable` events could get both a `stop` and a `reduce_to` — added a
  mutual-exclusion guard.
- Pending **credit** events were counted before settlement (8 "Pending merchant refund"
  rows) — excluded them.

**Progress:** amount_safe_to_pay 0.08→0.12, affordability_status 0.68→0.72,
recommended_payment_method 0.72→0.76 on the 25 solved samples.

---

## 6. Decision engine rewrite — rank every eligible safe plan, not first-match

**Prompt:**
> "go ahead with decision fix"

**Change:** replaced the fixed sequential order (full→installments→partial→wait, first match
wins) with full candidate enumeration — plain full, full+spending-changes, every valid+safe
installment option, and the earliest safe partial split — then selected by sorting a single
tuple key implementing the spec's exact 6 criteria (complete by deadline → no spending
changes → minimize total paid → start earlier → fewer payments → lowest
`payment_option_id`). `wait` became a fallback only when no immediate plan is safe, matching
its distinct eligibility rule. This surfaced 5 legitimate `partial_payment` cases the old
code never reached.
---

## 7. Final precision pass on `amount_safe_to_pay` + packaging

**Two more dataset-wide bugs fixed:**
- **Foreign-currency conversion** was missing for all events except image-derived ones — 140
  events across 27 users were booked in a foreign currency (one user's actual $1,800/month
  payroll was being summed as if it were IDR 1,800 next to an IDR 20M/quarter rent). A general
  `_convert_to_home_currency` step was added for every event. Verified directly: the traced
  user's min 90-day balance went from −60.9M IDR to +953,799 IDR. `not_affordable` dropped
  from 82→63 on the full dataset.
- **Income projection** preferred the repeated (mode) amount over the raw latest point, so a
  one-cycle blip (a "reduced due to unpaid leave" month) didn't permanently lock in the wrong
  rate; a genuinely-new rate (raise / first job) still uses the latest confirmed figure. An
  experiment to fix a 91-day leave gap regressed the full set (not_affordable 74→97) and was
  **reverted after full-dataset testing**, not trusted against the sample alone.

**Packaging:** wrote `code/README.md` (setup, model/429 diagnostics, run flags, module→stage
table, known limitations), added `.env.example` (key name only), and built `code.zip`
(requirements, README, 01_PLANNING.md, all of `code/` minus cache, and
`code/evaluation/usage_report.md`).

---

## 8. Final run & submission posture

- **Final full-dataset run**: 250/250 rows, `verify_output()` passes, **19 real model calls**
  (16 OCR + 3 text), ~$0.064 total, ~4.8 avg tokens/request — matching the submitted
  `usage_report.md`.
- The highest-leverage remaining item is `amount_safe_to_pay` **forecast precision** (the
  solver's softest sample metric), driven by remaining 90-day forecasting edge cases rather
  than an architectural defect.

---

## 9. Post-build review & submission planning (Cline review sessions)

Some of the most useful back-and-forth happened *after* the build — architecture affirmation and
final-interview prep. Highlights worth keeping for the interview:

1. **Architecture-vs-rubric review (read-only).** Confirmed the 7-stage design meets every README
   requirement — runnable from the terminal, exact output schema/order, one row per `request_id`,
   deterministic, no hardcoded labels, secrets via env, and a real (not placeholder) `usage_report`.
   The only weak graded dimension is `amount_safe_to_pay` (and its downstream
   `earliest_date_for_full_payment`), identified as a **forecast-accuracy** issue rather than an
   independent defect — the field itself is computed correctly as `min(safe_balance)` over the 90-day
   window. (A follow-on high-effort agent re-ran the *real* evaluation and cautioned that the early
   "systematically over-optimistic" framing came from a skip-enrichment run; the real run shows
   per-case direction, not one systematic bias — worth stating precisely in the interview.)
2. **Frontend decision.** No live deployable is required — the grader scores the batch `output.csv`
   and never calls a service. The right artifact is a static judge-facing page, not a reimplementation
   or an API that could diverge from the shipped predictions.
3. **Usage-report correctness.** The graded report must correspond to the final full-dataset run
   (Mistral pipeline: 19 calls, 884 in / 315 out, ~$0.064). Development/engineering-session tokens
   are supplementary and belong to the chat transcript, not the graded report (kept as a clearly
   labeled appendix).
4. **Submission readiness.** Rebuild `code.zip` after any final code change; keep `output.csv` and
   `usage_report.md` from a single `--force-refresh` run; secret hygiene (never ship `.env`); verify
   the 250-row contract; submit with buffer before 2026-09-13 18:00 IST; upload the curated
   transcript as `chat_transcript`.

*Token note: this review/planning conversation added roughly ~50k tokens to the engineering session.*