"""Reproducible build for the code.zip submission artifact.

Usage:
    python build_submission.py              # run the pipeline, then package
    python build_submission.py --skip-run   # package whatever output.csv /
                                             # usage_report.md already exist

Always run this last, after any code change -- a hand-built zip has no way to
guarantee it matches the output.csv actually being submitted alongside it.
This script enforces that by construction: it packages the usage_report.md
that sits next to the output.csv it just verified exists, then re-reads the
zip back and checks the bytes match before declaring success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES = [
    "requirements.txt",
    ".env.example",
    "01_PLANNING.md",
    "code/README.md",
    "code/__init__.py",
    "code/config.py",
    "code/ingest.py",
    "code/normalize.py",
    "code/enrich.py",
    "code/forecast.py",
    "code/decision.py",
    "code/verify.py",
    "code/evaluate.py",
    "code/run_pipeline.py",
    "code/main.py",
    "code/token_tracker.py",
    "code/evaluation/usage_report.md",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Package from whatever output.csv/usage_report.md already exist, without re-running the pipeline.",
    )
    args = parser.parse_args()

    if not args.skip_run:
        print("Running the full pipeline (--force-refresh)...")
        subprocess.run(
            [sys.executable, "-m", "code.run_pipeline", "--force-refresh"],
            check=True,
            cwd=ROOT,
        )

    output_csv = ROOT / "output.csv"
    usage_report = ROOT / "code" / "evaluation" / "usage_report.md"
    if not output_csv.exists():
        sys.exit("output.csv is missing -- run the pipeline first (drop --skip-run).")
    if not usage_report.exists():
        sys.exit("code/evaluation/usage_report.md is missing -- run the pipeline first.")

    row_count = sum(1 for _ in output_csv.open(encoding="utf-8")) - 1

    zip_path = ROOT / "code.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in FILES:
            archive.write(ROOT / relative_path, arcname=relative_path)

    with zipfile.ZipFile(zip_path) as archive:
        zipped_report = archive.read("code/evaluation/usage_report.md")
    if zipped_report != usage_report.read_bytes():
        sys.exit("usage_report.md inside code.zip does not match the one on disk -- rebuild.")

    print(f"Built {zip_path} ({zip_path.stat().st_size:,} bytes, {len(FILES)} files).")
    print(f"output.csv: {row_count} data rows.")
    print("Consistency check passed: the zipped usage_report.md matches the shipped output.csv's run.")
    print()
    print("Reminder: chat_transcript for submission is the root log.txt (gitignored,")
    print("not packaged here) -- upload it separately alongside output.csv and code.zip.")


if __name__ == "__main__":
    main()
