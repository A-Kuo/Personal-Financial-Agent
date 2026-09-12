from __future__ import annotations

import argparse
import time

import pandas as pd

from code.config import OUTPUT_COLUMNS, OUTPUT_PATH
from code.decision import make_decision
from code.enrich import extract_images, extract_messages, summarize_image_anomalies
from code.ingest import load_data
from code.normalize import normalize_events
from code.token_tracker import TokenTracker
from code.verify import verify_output


def generate_predictions(
    requests_override: pd.DataFrame | None = None,
    write_output: bool = True,
    skip_enrichment: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """skip_enrichment bypasses real Mistral calls (image_results/message_results
    come back empty) so the deterministic stages 4-6 can be smoke-tested without
    spending API tokens. Do not use it to generate a submitted output.csv.

    force_refresh re-calls the model even for already-cached extractions, so the
    usage report reflects real per-call token/cost numbers instead of $0 cache
    hits -- use this for the actual final submission run."""
    data = load_data()
    requests = requests_override.copy() if requests_override is not None else data.requests.copy()

    tracker = TokenTracker()
    if skip_enrichment:
        image_results = {}
        message_results = {}
    else:
        # usage_log.jsonl is append-only across processes, so without this a
        # report can pick up leftover entries from earlier/unrelated runs and
        # no longer match its own claim of "the final pipeline run." Extraction
        # result caches (image/message) are untouched -- only the spend log resets.
        tracker.reset()
        image_results = extract_images(data.images, tracker, force_refresh=force_refresh)
        # Free-tier safety: don't immediately switch endpoint families (OCR -> chat)
        # in case per-minute limits are shared across them on this key's tier.
        time.sleep(5)
        message_results = extract_messages(data.messages, tracker, force_refresh=force_refresh)

    normalized_events = normalize_events(
        events=data.events,
        profiles=data.profiles,
        exchange_rates=data.exchange_rates,
        image_results=image_results,
        message_results=message_results,
    )

    profiles = data.profiles.set_index("user_id")
    decisions = []
    for _, request in requests.iterrows():
        profile = profiles.loc[request["user_id"]]
        user_events = normalized_events[normalized_events["user_id"] == request["user_id"]].copy()
        user_options = data.payment_options[data.payment_options["request_id"] == request["request_id"]].copy()
        decisions.append(make_decision(request, profile, user_events, user_options).__dict__)

    output = pd.DataFrame(decisions, columns=OUTPUT_COLUMNS)
    verify_output(output, requests, data.events, data.payment_options)

    if write_output:
        output.to_csv(OUTPUT_PATH, index=False)
        anomalies = summarize_image_anomalies(data.images, image_results) if not skip_enrichment else []
        tracker.write_report(total_requests=len(output), anomalies=anomalies)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Bypass real Mistral calls for a structural dry run. Never use for a submitted output.csv.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-call the model even for cached extractions, so usage_report.md reflects real token/cost numbers. Use this for the final submission run.",
    )
    args = parser.parse_args()
    result = generate_predictions(
        write_output=not args.no_write,
        skip_enrichment=args.skip_enrichment,
        force_refresh=args.force_refresh,
    )
    print(f"Generated {len(result)} predictions.")
    if not args.no_write:
        print("Wrote output.csv and code/evaluation/usage_report.md")
