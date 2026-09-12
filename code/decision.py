from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from code.forecast import build_forecast, earliest_full_payment_date
from code.normalize import profile_preferences


@dataclass(frozen=True)
class Decision:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


def _fmt_amount(value: float) -> str:
    """Comma-formatted, for human-readable decision_explanation prose only."""
    return f"{round(float(value), 2):,.2f}".rstrip("0").rstrip(".")


def _fmt_plan_amount(value: float) -> str:
    """Plain numeric string for the machine-parseable payment_plan field -- no
    thousands separators, matching problem_statement.md's example format and
    sample_requests.csv (e.g. "15952906.67", never "15,952,906.67")."""
    return f"{round(float(value), 2):.2f}".rstrip("0").rstrip(".")


def _fmt_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _render_schedule(schedule: list[tuple[pd.Timestamp, float]]) -> str:
    return "|".join(f"{_fmt_date(date)}:{_fmt_plan_amount(amount)}" for date, amount in schedule)


def _installment_schedule(option: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    first = pd.Timestamp(option["first_payment_date"])
    amount = float(option["payment_amount"])
    frequency = int(option["payment_frequency_days"])
    count = int(option["number_of_payments"])
    return [(first + timedelta(days=frequency * i), amount) for i in range(count)]


def _valid_installment_option(option: pd.Series, profile: pd.Series) -> bool:
    prefs = profile_preferences(profile)
    if "installments" not in prefs["methods"]:
        return False
    maximum = profile.get("max_installment_months")
    if pd.isna(maximum):
        return False
    duration_months = int(option["number_of_payments"]) * int(option["payment_frequency_days"]) / 30
    return duration_months <= float(maximum) + 1e-9


def _flexible_candidates(profile: pd.Series, user_events: pd.DataFrame) -> list[dict]:
    prefs = profile_preferences(profile)
    events = user_events[user_events["flexibility"].isin({"stoppable", "reducible", "reducibleorstoppable"})].copy()
    events = events.sort_values("cash_date", ascending=False).drop_duplicates(
        subset=["description", "category", "flexibility"], keep="first"
    )
    candidates: list[dict] = []

    for row in events.itertuples(index=False):
        if row.category in prefs["protected"]:
            continue
        flexibility = str(row.flexibility)
        amount = float(row.amount)
        if "stoppable" in flexibility and row.category in prefs["stoppable"]:
            candidates.append({"event_id": row.event_id, "action": "stop", "amount": 0.0, "cash_freed": amount, "description": row.description})
        if "reducible" in flexibility and row.category in prefs["reducible"] and pd.notna(row.minimum_allowed_amount):
            floor = float(row.minimum_allowed_amount)
            candidates.append({"event_id": row.event_id, "action": "reduce_to", "amount": floor, "cash_freed": max(amount - floor, 0.0), "description": row.description})
    return sorted(candidates, key=lambda item: item["cash_freed"], reverse=True)


def _find_changes_for_full_payment(
    profile: pd.Series,
    user_events: pd.DataFrame,
    request_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    requested_amount: float,
) -> list[dict] | None:
    selected: list[dict] = []
    for candidate in _flexible_candidates(profile, user_events):
        selected.append(candidate)
        if len(selected) > 3:
            break
        forecast = build_forecast(
            profile, user_events, request_date, horizon_end,
            spending_changes=selected,
            payment_schedule=[(request_date, requested_amount)],
        )
        if (forecast["safe_balance"] >= -1e-6).all():
            return selected
    return None


def _render_changes(changes: list[dict]) -> str:
    if not changes:
        return "none"
    rendered = []
    for change in changes:
        if change["action"] == "stop":
            rendered.append(f"stop:{change['event_id']}")
        else:
            rendered.append(f"reduce_to:{change['event_id']}:{_fmt_plan_amount(change['amount'])}")
    return "|".join(rendered)


def make_decision(request: pd.Series, profile: pd.Series, user_events: pd.DataFrame, options: pd.DataFrame) -> Decision:
    request_date = pd.Timestamp(request["request_date"]).normalize()
    desired_date = pd.Timestamp(request["desired_completion_date"]).normalize()
    horizon_end = max(desired_date, request_date + timedelta(days=90))
    requested_amount = float(request["requested_amount"])
    currency = str(profile["home_currency"])
    minimum = float(profile["minimum_balance_to_keep"])

    prefs = profile_preferences(profile)
    full_payment_eligible = "full_payment" in prefs["methods"]

    baseline = build_forecast(profile, user_events, request_date, horizon_end)
    # The most payable today without breaking the 90-day safety check is bounded
    # by the LOWEST safe_balance anywhere in the forecast, not just today's --
    # paying X today reduces every later day's running balance by the same X, so
    # a future dip smaller than today's headroom caps what's actually safe now.
    # (baseline.iloc[0] alone under-constrains this: request_79 in the sample
    # dataset has 1104 safe today but only 791 at its day-9 low, and using the
    # day-0 figure let a partial-payment first installment retroactively break
    # that later day, which showed up as a false not_affordable/verify failure.)
    safe_today = max(0.0, float(baseline["safe_balance"].min()))
    # Capacity to pay the full amount as a single payment, independent of which
    # method the user actually accepts -- see problem_statement.md's note that
    # this field "may equal request_date even when the selected recommendation
    # is installments because the user has chosen not to consider full payment."
    capacity_earliest = earliest_full_payment_date(baseline, requested_amount)
    full_schedule = [(request_date, requested_amount)]
    full_forecast = build_forecast(profile, user_events, request_date, horizon_end, payment_schedule=full_schedule)

    if full_payment_eligible and (full_forecast["safe_balance"] >= -1e-6).all():
        return Decision(request["request_id"], requested_amount, "affordable_now", "full_payment", _render_schedule(full_schedule), _fmt_date(request_date), "none", f"Pay {currency} {_fmt_amount(requested_amount)} today. This keeps the {currency} {_fmt_amount(minimum)} minimum protected over the next 90 days.")

    changes = _find_changes_for_full_payment(profile, user_events, request_date, horizon_end, requested_amount) if full_payment_eligible else None
    if changes:
        changed_forecast = build_forecast(profile, user_events, request_date, horizon_end, spending_changes=changes, payment_schedule=full_schedule)
        if (changed_forecast["safe_balance"] >= -1e-6).all():
            first = changes[0]
            action = first["action"].replace("_", " ").capitalize()
            return Decision(request["request_id"], requested_amount, "affordable_with_plan", "full_payment", _render_schedule(full_schedule), _fmt_date(request_date), _render_changes(changes), f"{action} {first['description']}, then pay {currency} {_fmt_amount(requested_amount)} today. This keeps the {currency} {_fmt_amount(minimum)} minimum protected.")

    installments = options[(options["request_id"] == request["request_id"]) & (options["payment_method"] == "installments")].copy()
    if not installments.empty:
        installments = installments[installments.apply(_valid_installment_option, axis=1, profile=profile)].sort_values(["financing_fee", "total_payable_amount"])
        for _, option in installments.iterrows():
            schedule = _installment_schedule(option)
            if schedule[-1][0] > horizon_end:
                continue
            forecast = build_forecast(profile, user_events, request_date, horizon_end, payment_schedule=schedule)
            if (forecast["safe_balance"] >= -1e-6).all():
                first_payment = schedule[0][1]
                earliest_full_text = _fmt_date(capacity_earliest) if capacity_earliest is not None else ""
                return Decision(request["request_id"], min(first_payment, requested_amount), "affordable_with_plan", "installments", _render_schedule(schedule), earliest_full_text, "none", f"Use {int(option['number_of_payments'])} installments of {currency} {_fmt_amount(first_payment)}, starting {_fmt_date(schedule[0][0])}. This keeps the {currency} {_fmt_amount(minimum)} minimum protected.")

    if bool(request["allows_partial_payment"]) and "partial_payment" in prefs["methods"] and safe_today > 0:
        first_payment = min(safe_today, requested_amount - 0.01)
        remainder = requested_amount - first_payment
        for _, row in baseline.iterrows():
            date = pd.Timestamp(row["date"])
            if not request_date < date <= desired_date:
                continue
            schedule = [(request_date, first_payment), (date, remainder)]
            forecast = build_forecast(profile, user_events, request_date, horizon_end, payment_schedule=schedule)
            if (forecast["safe_balance"] >= -1e-6).all():
                return Decision(request["request_id"], first_payment, "affordable_with_plan", "partial_payment", _render_schedule(schedule), _fmt_date(date), "none", f"Pay {currency} {_fmt_amount(first_payment)} today and the remaining {currency} {_fmt_amount(remainder)} on {_fmt_date(date)}. This completes the full request and keeps the {currency} {_fmt_amount(minimum)} minimum protected.")

    if full_payment_eligible and capacity_earliest is not None:
        return Decision(request["request_id"], min(safe_today, requested_amount), "affordable_later", "wait", _render_schedule([(capacity_earliest, requested_amount)]), _fmt_date(capacity_earliest), "none", f"Pay {currency} {_fmt_amount(requested_amount)} in full on {_fmt_date(capacity_earliest)}. Paying earlier would put the {currency} {_fmt_amount(minimum)} minimum at risk.")

    return Decision(request["request_id"], min(safe_today, requested_amount), "not_affordable", "not_recommended", "none", "", "none", f"Do not make this payment by {_fmt_date(desired_date)}. None of the available options keeps the {currency} {_fmt_amount(minimum)} minimum protected.")
