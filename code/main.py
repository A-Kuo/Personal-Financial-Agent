"""Entry point documented in README.md: `python3 code/main.py`.

Bootstraps the repo root onto sys.path so the `code.*` absolute imports used
throughout this package resolve when this file is run directly rather than
via `python -m code.run_pipeline`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.run_pipeline import generate_predictions  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    result = generate_predictions(skip_enrichment=args.skip_enrichment, force_refresh=args.force_refresh)
    print(f"Generated {len(result)} predictions.")
    print("Wrote output.csv and code/evaluation/usage_report.md")
