from __future__ import annotations

from datetime import timedelta

import pandas as pd

from code.config import OUTPUT_COLUMNS, VALID_PAYMENT_METHODS, VALID_STATUSES


def _parse_plan(plan: str) -> list[tuple[pd.Timestamp, float]]:
    if not plan or str(plan).strip().lower() == "none":
        return []
    results = []
    for item in str(plan).split("|"):
        date_text, amount_text = item.split(":", maxsplit=1)
        results.append((pd.Timestamp(date_text), float(amount_text.replace(",", ""))))
    return results


def _render_option_plan(option: pd.Series) -> str:
    first = pd.Timestamp(option["first_payment_date"])
    amount = float(option["payment_amount"])
    frequency = int(option["payment_frequency_days"])
    count = int(option["number_of_payments"])
    fmt = lambda value: f"{round(value, 2):.2f}".rstrip("0").rstrip(".")
    return "|".join(
        f"{(first + timedelta(days=frequency * i)).strftime('%Y-%m-%d')}:{fmt(amount)}"
        for i in range(count)
    )


def verify_output(
    output: pd.DataFrame,
    requests: pd.DataFrame,
    events: pd.DataFrame,
    payment_options: pd.DataFrame,
) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError("Output columns are not in the required order.")
    if len(output) != len(requests):
        raise ValueError(f"Expected {len(requests)} output rows; found {len(output)}.")
    if output["request_id"].duplicated().any():
        raise ValueError("output.csv contains duplicate request_id values.")
    if set(output["request_id"]) != set(requests["request_id"]):
        raise ValueError("output.csv request IDs do not match requests.csv.")

    requested = requests.set_index("request_id")["requested_amount"].to_dict()
    valid_installment_plans = {
        (row.request_id, _render_option_plan(pd.Series(row._asdict())))
        for row in payment_options.itertuples(index=False)
        if row.payment_method == "installments"
    }
    flexibility = events.set_index("event_id")["flexibility"].to_dict()

    for row in output.itertuples(index=False):
        amount = float(row.amount_safe_to_pay)
        if not 0 <= amount <= float(requested[row.request_id]) + 1e-6:
            raise ValueError(f"{row.request_id}: amount_safe_to_pay is out of bounds.")
        if row.affordability_status not in VALID_STATUSES:
            raise ValueError(f"{row.request_id}: invalid affordability status.")
        if row.recommended_payment_method not in VALID_PAYMENT_METHODS:
            raise ValueError(f"{row.request_id}: invalid payment method.")

        schedule = _parse_plan(row.payment_plan)
        if row.recommended_payment_method == "installments" and (row.request_id, row.payment_plan) not in valid_installment_plans:
            raise ValueError(f"{row.request_id}: installment plan does not exactly match an offered option.")
        if row.recommended_payment_method == "partial_payment":
            if len(schedule) != 2:
                raise ValueError(f"{row.request_id}: partial payment must have exactly two entries.")
            if abs(sum(value for _, value in schedule) - float(requested[row.request_id])) > 0.02:
                raise ValueError(f"{row.request_id}: partial plan does not sum to the request amount.")

        changes = str(row.spending_changes_needed)
        if changes != "none":
            entries = changes.split("|")
            if len(entries) > 3:
                raise ValueError(f"{row.request_id}: more than three spending changes.")
            for entry in entries:
                parts = entry.split(":")
                if len(parts) < 2 or parts[0] not in {"stop", "reduce_to"}:
                    raise ValueError(f"{row.request_id}: invalid spending change format.")
                event_id = parts[1]
                event_flexibility = str(flexibility.get(event_id, ""))
                if not event_flexibility or event_flexibility == "fixed":
                    raise ValueError(f"{row.request_id}: change targets fixed/missing event {event_id}.")
                if parts[0] == "stop" and "stoppable" not in event_flexibility:
                    raise ValueError(f"{row.request_id}: non-stoppable event cannot be stopped.")
                if parts[0] == "reduce_to" and "reducible" not in event_flexibility:
                    raise ValueError(f"{row.request_id}: non-reducible event cannot be reduced.")

        if row.affordability_status == "not_affordable":
            if pd.notna(row.earliest_date_for_full_payment) and str(row.earliest_date_for_full_payment).strip():
                raise ValueError(f"{row.request_id}: not_affordable must have blank earliest date.")
        elif not str(row.earliest_date_for_full_payment).strip():
            raise ValueError(f"{row.request_id}: a feasible outcome requires an earliest payment date.")
