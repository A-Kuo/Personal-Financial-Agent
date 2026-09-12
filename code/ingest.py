from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from code.config import DATASET_DIR, OUTPUT_COLUMNS


@dataclass(frozen=True)
class DataBundle:
    requests: pd.DataFrame
    sample_requests: pd.DataFrame
    profiles: pd.DataFrame
    events: pd.DataFrame
    payment_options: pd.DataFrame
    exchange_rates: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    return pd.read_csv(path)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _validate_inputs(bundle: DataBundle) -> None:
    required_request_cols = {
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
    }
    required_event_cols = {
        "event_id",
        "user_id",
        "event_type",
        "description",
        "category",
        "direction",
        "amount",
        "currency",
        "event_date",
        "settlement_date",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    }

    missing_requests = required_request_cols.difference(bundle.requests.columns)
    missing_events = required_event_cols.difference(bundle.events.columns)

    if missing_requests:
        raise ValueError(f"requests.csv missing columns: {sorted(missing_requests)}")
    if missing_events:
        raise ValueError(f"financial_events.csv missing columns: {sorted(missing_events)}")
    if bundle.requests["request_id"].duplicated().any():
        raise ValueError("requests.csv contains duplicate request_id values.")
    if bundle.profiles["user_id"].duplicated().any():
        raise ValueError("financial_profiles.csv contains duplicate user_id values.")

    required_output = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ]
    if required_output != OUTPUT_COLUMNS:
        raise ValueError("Configured output schema does not match required schema.")

    missing_profiles = set(bundle.requests["user_id"]) - set(bundle.profiles["user_id"])
    if missing_profiles:
        raise ValueError(f"Missing profiles for request users: {sorted(missing_profiles)}")


def load_data(dataset_dir: Path = DATASET_DIR) -> DataBundle:
    requests = _read_csv(dataset_dir / "requests.csv")
    sample_requests = _read_csv(dataset_dir / "sample_requests.csv")
    profiles = _read_csv(dataset_dir / "financial_profiles.csv")
    events = _read_csv(dataset_dir / "financial_events.csv")
    payment_options = _read_csv(dataset_dir / "request_payment_options.csv")
    exchange_rates = _read_csv(dataset_dir / "exchange_rates.csv")
    messages = _read_csv(dataset_dir / "messages.csv")
    images = _read_csv(dataset_dir / "images.csv")

    requests["allows_partial_payment"] = requests["allows_partial_payment"].map(_to_bool)
    sample_requests["allows_partial_payment"] = sample_requests["allows_partial_payment"].map(
        _to_bool
    )

    requests = _parse_dates(requests, ["request_date", "desired_completion_date"])
    sample_requests = _parse_dates(
        sample_requests,
        ["request_date", "desired_completion_date", "earliest_date_for_full_payment"],
    )
    events = _parse_dates(events, ["event_date", "settlement_date"])
    payment_options = _parse_dates(payment_options, ["first_payment_date"])
    exchange_rates = _parse_dates(exchange_rates, ["rate_date"])
    messages = _parse_dates(messages, ["sent_at"])

    for frame in (requests, sample_requests):
        frame["requested_amount"] = pd.to_numeric(frame["requested_amount"], errors="raise")

    for column in ("current_available_balance", "minimum_balance_to_keep"):
        profiles[column] = pd.to_numeric(profiles[column], errors="raise")

    for column in ("amount", "minimum_allowed_amount"):
        events[column] = pd.to_numeric(events[column], errors="coerce")

    for column in (
        "payment_amount",
        "number_of_payments",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
    ):
        payment_options[column] = pd.to_numeric(payment_options[column], errors="coerce")

    exchange_rates["rate"] = pd.to_numeric(exchange_rates["rate"], errors="raise")

    bundle = DataBundle(
        requests=requests,
        sample_requests=sample_requests,
        profiles=profiles,
        events=events,
        payment_options=payment_options,
        exchange_rates=exchange_rates,
        messages=messages,
        images=images,
    )
    _validate_inputs(bundle)
    return bundle
