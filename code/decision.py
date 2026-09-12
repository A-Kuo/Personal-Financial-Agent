from __future__ import annotations

import re
from dataclasses import dataclass, field
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


def _option_sort_number(payment_option_id: object) -> float:
    """Extract the numeric suffix for a correct numeric (not lexicographic)
    comparison of payment_option_id -- "payment_option_10" must sort after
    "payment_option_9", which plain string comparison gets wrong."""
    match = re.search(r"(\d+)$", str(payment_option_id))
    return float(match.group(1)) if match else float("inf")


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
    used_event_ids: set[str] = set()
    for candidate in _flexible_candidates(profile, user_events):
        # A "reducibleorstoppable" event yields both a stop and a reduce_to
        # candidate; stopping and reducing the same event are mutually
        # exclusive, so once either is selected for an event, skip the other.
        if str(candidate["event_id"]) in used_event_ids:
            continue
        selected.append(candidate)
        used_event_ids.add(str(candidate["event_id"]))
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


@dataclass(frozen=True)
class Candidate:
    """One eligible, already-verified-safe immediate plan. `wait` is not a
    Candidate -- it's a separate fallback evaluated only when no Candidate
    exists at all, per problem_statement.md's eligibility rules."""

    method: str  # "full_payment" | "installments" | "partial_payment"
    schedule: list[tuple[pd.Timestamp, float]]
    spending_changes: list[dict] = field(default_factory=list)
    total_paid: float = 0.0
    option_number: float = float("inf")
    payment_option_id: str = ""
    explanation: str = ""

    def rank_key(self, desired_date: pd.Timestamp) -> tuple:
        # Exactly the 6 criteria in problem_statement.md's "Choosing Between
        # Safe Plans", in order -- lower tuples win.
        completes_by_deadline = self.schedule[-1][0] <= desired_date
        return (
            0 if completes_by_deadline else 1,  # 1. complete by desired_completion_date
            len(self.spending_changes),  # 2. require no spending changes
            round(self.total_paid, 6),  # 3. minimize the total amount paid
            self.schedule[0][0],  # 4. start payment earlier
            len(self.schedule),  # 5. use fewer payments
            self.option_number,  # 6. lowest payment_option_id as final tie-breaker
        )


def _build_candidates(
    request: pd.Series,
    profile: pd.Series,
    user_events: pd.DataFrame,
    options: pd.DataFrame,
    request_date: pd.Timestamp,
    desired_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    requested_amount: float,
    amount_safe_to_pay: float,
    full_payment_eligible: bool,
    prefs: dict[str, set[str]],
    currency: str,
    minimum: float,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    if full_payment_eligible:
        full_schedule = [(request_date, requested_amount)]
        full_forecast = build_forecast(profile, user_events, request_date, horizon_end, payment_schedule=full_schedule)
        if (full_forecast["safe_balance"] >= -1e-6).all():
            candidates.append(Candidate(
                "full_payment", full_schedule, [], requested_amount, explanation=(
                    f"Pay {currency} {_fmt_amount(requested_amount)} today. This keeps the "
                    f"{currency} {_fmt_amount(minimum)} minimum protected over the next 90 days."
                ),
            ))
        else:
            changes = _find_changes_for_full_payment(profile, user_events, request_date, horizon_end, requested_amount)
            if changes:
                changed_forecast = build_forecast(
                    profile, user_events, request_date, horizon_end,
                    spending_changes=changes, payment_schedule=full_schedule,
                )
                if (changed_forecast["safe_balance"] >= -1e-6).all():
                    first = changes[0]
                    action = first["action"].replace("_", " ").capitalize()
                    candidates.append(Candidate(
                        "full_payment", full_schedule, changes, requested_amount, explanation=(
                            f"{action} {first['description']}, then pay {currency} {_fmt_amount(requested_amount)} "
                            f"today. This keeps the {currency} {_fmt_amount(minimum)} minimum protected."
                        ),
                    ))

    installments = options[(options["request_id"] == request["request_id"]) & (options["payment_method"] == "installments")].copy()
    if not installments.empty:
        installments = installments[installments.apply(_valid_installment_option, axis=1, profile=profile)]
        for _, option in installments.iterrows():
            schedule = _installment_schedule(option)
            forecast = build_forecast(profile, user_events, request_date, horizon_end, payment_schedule=schedule)
            if (forecast["safe_balance"] >= -1e-6).all():
                first_payment = schedule[0][1]
                candidates.append(Candidate(
                    "installments", schedule, [],
                    total_paid=float(option["total_payable_amount"]),
                    option_number=_option_sort_number(option["payment_option_id"]),
                    payment_option_id=str(option["payment_option_id"]),
                    explanation=(
                        f"Use {int(option['number_of_payments'])} installments of {currency} {_fmt_amount(first_payment)}, "
                        f"starting {_fmt_date(schedule[0][0])}. This keeps the {currency} {_fmt_amount(minimum)} minimum protected."
                    ),
                ))

    if bool(request["allows_partial_payment"]) and "partial_payment" in prefs["methods"] and 0 < amount_safe_to_pay < requested_amount:
        first_payment = amount_safe_to_pay
        remainder = requested_amount - first_payment
        baseline_dates = build_forecast(profile, user_events, request_date, horizon_end)
        for _, row in baseline_dates.iterrows():
            date = pd.Timestamp(row["date"])
            if not request_date < date <= desired_date:
                continue
            schedule = [(request_date, first_payment), (date, remainder)]
            forecast = build_forecast(profile, user_events, request_date, horizon_end, payment_schedule=schedule)
            if (forecast["safe_balance"] >= -1e-6).all():
                candidates.append(Candidate(
                    "partial_payment", schedule, [], requested_amount, explanation=(
                        f"Pay {currency} {_fmt_amount(first_payment)} today and the remaining {currency} "
                        f"{_fmt_amount(remainder)} on {_fmt_date(date)}. This completes the full request and "
                        f"keeps the {currency} {_fmt_amount(minimum)} minimum protected."
                    ),
                ))
                break  # earliest safe remainder date; all other criteria tie among partial's own options

    return candidates


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
    safe_today = max(0.0, float(baseline["safe_balance"].min()))
    # amount_safe_to_pay is capacity "before optional spending changes",
    # independent of the chosen method/plan -- confirmed against
    # sample_requests.csv: request_06 pays its full 620.40 via a spending change
    # but reports amount_safe_to_pay=603.30 (the pre-change baseline), and
    # request_12 reports amount_safe_to_pay == requested_amount while still being
    # recommended installments (the user's accepted methods just exclude
    # full_payment). One baseline figure, used regardless of which plan wins.
    amount_safe_to_pay = min(safe_today, requested_amount)
    # Capacity to pay the full amount as a single payment, independent of which
    # method the user actually accepts -- problem_statement.md notes this field
    # "may equal request_date even when the selected recommendation is
    # installments because the user has chosen not to consider full payment."
    capacity_earliest = earliest_full_payment_date(baseline, requested_amount)

    candidates = _build_candidates(
        request, profile, user_events, options, request_date, desired_date, horizon_end,
        requested_amount, amount_safe_to_pay, full_payment_eligible, prefs, currency, minimum,
    )

    if candidates:
        winner = min(candidates, key=lambda candidate: candidate.rank_key(desired_date))
        is_plain_full_payment_today = winner.method == "full_payment" and not winner.spending_changes
        status = "affordable_now" if is_plain_full_payment_today else "affordable_with_plan"
        earliest_text = {
            "full_payment": _fmt_date(request_date),
            "partial_payment": _fmt_date(winner.schedule[-1][0]),
            "installments": _fmt_date(capacity_earliest) if capacity_earliest is not None else "",
        }[winner.method]
        return Decision(
            request["request_id"], amount_safe_to_pay, status, winner.method,
            _render_schedule(winner.schedule), earliest_text,
            _render_changes(winner.spending_changes), winner.explanation,
        )

    if full_payment_eligible and capacity_earliest is not None:
        return Decision(request["request_id"], amount_safe_to_pay, "affordable_later", "wait", _render_schedule([(capacity_earliest, requested_amount)]), _fmt_date(capacity_earliest), "none", f"Pay {currency} {_fmt_amount(requested_amount)} in full on {_fmt_date(capacity_earliest)}. Paying earlier would put the {currency} {_fmt_amount(minimum)} minimum at risk.")

    return Decision(request["request_id"], amount_safe_to_pay, "not_affordable", "not_recommended", "none", "", "none", f"Do not make this payment by {_fmt_date(desired_date)}. None of the available options keeps the {currency} {_fmt_amount(minimum)} minimum protected.")
