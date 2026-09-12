from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd


def _infer_frequency_days(group: pd.DataFrame) -> int | None:
    dates = sorted(pd.to_datetime(group["cash_date"].dropna()).unique())
    if len(dates) < 2:
        return None
    diffs = np.diff(np.array(dates, dtype="datetime64[D]")).astype(int)
    median = float(np.median(diffs))
    if 25 <= median <= 35:
        return 30
    if 12 <= median <= 16:
        return 14
    if 6 <= median <= 8:
        return 7
    if 80 <= median <= 100:
        return 91
    return None


def _project_group(group: pd.DataFrame, horizon_end: pd.Timestamp, use_latest_amount: bool) -> list[dict]:
    frequency = _infer_frequency_days(group)
    if frequency is None:
        return []
    latest = group.sort_values("cash_date").iloc[-1]
    if use_latest_amount:
        # Conservative for income: project forward at the most recently confirmed
        # rate rather than blending in a one-off prorated-first-payment or bonus
        # that isn't representative of the ongoing recurring amount.
        typical_amount = float(latest["amount"])
    else:
        typical_amount = float(group["amount"].tail(min(4, len(group))).median())
    rows: list[dict] = []
    next_date = pd.Timestamp(latest["cash_date"]) + timedelta(days=frequency)
    while next_date <= horizon_end:
        projected = latest.to_dict()
        projected["event_id"] = f"projected::{latest['event_id']}::{next_date.date()}"
        projected["cash_date"] = next_date
        projected["amount"] = typical_amount
        projected["signed_amount"] = typical_amount if projected["direction"] == "credit" else -typical_amount
        projected["status"] = "projected"
        projected["projected"] = True
        rows.append(projected)
        next_date += timedelta(days=frequency)
    return rows


def _project_recurring_events(
    history: pd.DataFrame,
    request_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
) -> pd.DataFrame:
    # Known real data points -- historical AND already-confirmed/scheduled future
    # ones -- both inform the cadence and anchor where projection should resume.
    # Using only cash_date <= request_date (as before) excludes an already-known
    # "next confirmed salary" from ever being recognized, which either starves
    # income projection entirely (sparse-history users) or, when income history
    # IS rich enough to self-detect a cadence under a different description (e.g.
    # "Payroll credit" vs "Next confirmed salary" for the same job), lets the
    # projector fabricate a duplicate payday right next to the real confirmed one.
    known = history[history["cash_date"] <= horizon_end].copy()
    rows: list[dict] = []

    income = known[known["event_type"] == "income"]
    # Grouped by category rather than description: real payroll data relabels
    # the next confirmed payment differently from prior "Payroll credit" rows,
    # so description-based grouping treats them as unrelated series.
    for _, group in income.groupby(["user_id", "category", "direction", "currency"], dropna=False):
        rows.extend(_project_group(group, horizon_end, use_latest_amount=True))

    other = known[known["event_type"] != "income"]
    group_cols = ["user_id", "event_type", "description", "category", "direction", "currency"]
    for _, group in other.groupby(group_cols, dropna=False):
        rows.extend(_project_group(group, horizon_end, use_latest_amount=False))

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=history.columns)


def build_forecast(
    profile: pd.Series,
    user_events: pd.DataFrame,
    request_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    spending_changes: list[dict] | None = None,
    payment_schedule: list[tuple[pd.Timestamp, float]] | None = None,
) -> pd.DataFrame:
    spending_changes = spending_changes or []
    payment_schedule = payment_schedule or []

    events = user_events[user_events["confirmed_by_message"]].copy()
    future_actual = events[
        (events["cash_date"] >= request_date)
        & (events["cash_date"] <= horizon_end)
        # "Do not count pending credits ... until they settle" -- a pending
        # refund/bonus/commission isn't cash yet even if it's dated within the
        # forecast window. Pending debits are handled separately (reserved).
        & ~((events["status"] == "pending") & (events["direction"] == "credit"))
    ].copy()
    projected = _project_recurring_events(events, request_date, horizon_end)
    all_events = pd.concat([future_actual, projected], ignore_index=True, sort=False)

    changes = {str(change["event_id"]): change for change in spending_changes}
    for index, event in all_events.iterrows():
        event_id = str(event["event_id"])
        base_id = event_id.split("::")[1] if event_id.startswith("projected::") else event_id
        change = changes.get(base_id)
        if change is None or event["direction"] != "debit":
            continue
        if change["action"] == "stop":
            all_events.loc[index, "signed_amount"] = 0.0
        elif change["action"] == "reduce_to":
            all_events.loc[index, "signed_amount"] = -float(change["amount"])

    daily_cash = defaultdict(float)
    for event in all_events.itertuples(index=False):
        if pd.notna(event.cash_date):
            daily_cash[pd.Timestamp(event.cash_date).normalize()] += float(event.signed_amount)
    for date, amount in payment_schedule:
        daily_cash[pd.Timestamp(date).normalize()] -= float(amount)

    pending_debits = events[
        (events["status"] == "pending")
        & (events["direction"] == "debit")
        & (events["cash_date"] < request_date)
    ]
    pending_reserve = float(pending_debits["amount"].sum())

    balance = float(profile["current_available_balance"]) - pending_reserve
    minimum = float(profile["minimum_balance_to_keep"])
    dates = pd.date_range(request_date.normalize(), horizon_end.normalize(), freq="D")
    results: list[dict] = []
    for date in dates:
        balance += daily_cash[date]
        results.append(
            {"date": date, "projected_balance": balance, "safe_balance": balance - minimum}
        )
    return pd.DataFrame(results)


def earliest_full_payment_date(forecast: pd.DataFrame, requested_amount: float) -> pd.Timestamp | None:
    eligible = forecast[forecast["safe_balance"] >= requested_amount]
    if eligible.empty:
        return None
    return pd.Timestamp(eligible.iloc[0]["date"])
