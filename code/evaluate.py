from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from code.run_pipeline import generate_predictions


def _normal_plan(value: object) -> list[tuple[str, float]]:
    if pd.isna(value) or str(value).strip().lower() in {"", "none"}:
        return []
    return [
        (date, round(float(amount.replace(",", "")), 2))
        for date, amount in (entry.split(":", maxsplit=1) for entry in str(value).split("|"))
    ]


def _normal_changes(value: object) -> set[str]:
    if pd.isna(value) or str(value).strip().lower() in {"", "none"}:
        return set()
    return set(str(value).split("|"))


def evaluate(dataset_dir: Path, skip_enrichment: bool = False) -> pd.DataFrame:
    sample = pd.read_csv(dataset_dir / "sample_requests.csv")
    input_columns = [
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
    ]
    predictions = generate_predictions(
        requests_override=sample[input_columns], write_output=False, skip_enrichment=skip_enrichment
    )
    merged = sample.merge(predictions, on="request_id", suffixes=("_expected", "_predicted"))

    expected = pd.to_numeric(merged["amount_safe_to_pay_expected"], errors="coerce")
    predicted = pd.to_numeric(merged["amount_safe_to_pay_predicted"], errors="coerce")
    numeric_match = (expected - predicted).abs() <= expected.abs().clip(lower=1.0) * 0.005

    results = [{
        "field": "amount_safe_to_pay",
        "accuracy": float(numeric_match.mean()),
        "matches": int(numeric_match.sum()),
        "total": len(merged),
    }]

    for field in ("affordability_status", "recommended_payment_method", "earliest_date_for_full_payment"):
        match = merged[f"{field}_expected"].fillna("").astype(str) == merged[f"{field}_predicted"].fillna("").astype(str)
        results.append({"field": field, "accuracy": float(match.mean()), "matches": int(match.sum()), "total": len(match)})

    plan_match = merged.apply(lambda row: _normal_plan(row["payment_plan_expected"]) == _normal_plan(row["payment_plan_predicted"]), axis=1)
    results.append({"field": "payment_plan", "accuracy": float(plan_match.mean()), "matches": int(plan_match.sum()), "total": len(plan_match)})

    change_match = merged.apply(lambda row: _normal_changes(row["spending_changes_needed_expected"]) == _normal_changes(row["spending_changes_needed_predicted"]), axis=1)
    results.append({"field": "spending_changes_needed", "accuracy": float(change_match.mean()), "matches": int(change_match.sum()), "total": len(change_match)})

    result_df = pd.DataFrame(results)
    print(result_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    mismatches = merged.loc[~numeric_match, [
        "request_id",
        "amount_safe_to_pay_expected",
        "amount_safe_to_pay_predicted",
        "affordability_status_expected",
        "affordability_status_predicted",
        "recommended_payment_method_expected",
        "recommended_payment_method_predicted",
    ]]
    if not mismatches.empty:
        print("\nAmount mismatches:")
        print(mismatches.to_string(index=False))
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Bypass real Mistral calls for a structural dry run. Never use for a submitted score.",
    )
    args = parser.parse_args()
    evaluate(Path(args.dataset_dir), skip_enrichment=args.skip_enrichment)
