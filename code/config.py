from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"
CACHE_DIR = ROOT_DIR / "code" / "cache"
EVALUATION_DIR = ROOT_DIR / "code" / "evaluation"

OUTPUT_PATH = ROOT_DIR / "output.csv"
USAGE_LOG_PATH = EVALUATION_DIR / "usage_log.jsonl"
USAGE_REPORT_PATH = EVALUATION_DIR / "usage_report.md"
IMAGE_CACHE_PATH = CACHE_DIR / "image_extractions.json"
MESSAGE_CACHE_PATH = CACHE_DIR / "message_extractions.json"

FORECAST_DAYS = 90
MAX_SPENDING_CHANGES = 3

OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

VALID_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}

VALID_PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}
