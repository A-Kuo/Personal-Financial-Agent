from __future__ import annotations

from collections import deque

import pandas as pd

from code.enrich import ImageExtraction, MessageExtraction

CASH_EVENT_TYPES = {
    "expense",
    "income",
    "subscription",
    "debtpayment",
    "investmentpurchase",
    "refund",
}
EXCLUDED_EVENT_TYPES = {"investmentvaluation"}
UNCONFIRMED_MESSAGE_TYPES = {
    "unconfirmed_income",
    "refund_pending",
    "investment_non_cash",
    "scam_or_advance_fee",
}


def _split_pipe(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {token.strip() for token in str(value).split("|") if token.strip()}


def _nearest_rate(
    rates: pd.DataFrame,
    from_currency: str,
    to_currency: str,
    date: pd.Timestamp,
) -> float | None:
    direct = rates[
        (rates["from_currency"] == from_currency)
        & (rates["to_currency"] == to_currency)
    ].copy()
    if not direct.empty:
        prior = direct[direct["rate_date"] <= date]
        if not prior.empty:
            return float(prior.sort_values("rate_date").iloc[-1]["rate"])
        direct["distance"] = (direct["rate_date"] - date).abs()
        return float(direct.sort_values("distance").iloc[0]["rate"])

    inverse = rates[
        (rates["from_currency"] == to_currency)
        & (rates["to_currency"] == from_currency)
    ].copy()
    if not inverse.empty:
        prior = inverse[inverse["rate_date"] <= date]
        selected = (
            prior.sort_values("rate_date").iloc[-1]
            if not prior.empty
            else inverse.assign(distance=(inverse["rate_date"] - date).abs())
            .sort_values("distance")
            .iloc[0]
        )
        return 1.0 / float(selected["rate"])
    return None


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    date: pd.Timestamp,
    rates: pd.DataFrame,
) -> float | None:
    if from_currency == to_currency:
        return float(amount)

    currencies = set(rates["from_currency"]).union(set(rates["to_currency"]))
    graph: dict[str, set[str]] = {currency: set() for currency in currencies}
    for row in rates.itertuples(index=False):
        graph[row.from_currency].add(row.to_currency)
        graph[row.to_currency].add(row.from_currency)

    queue = deque([(from_currency, 1.0)])
    visited = {from_currency}
    while queue:
        current, multiplier = queue.popleft()
        if current == to_currency:
            return float(amount) * multiplier
        for neighbor in graph.get(current, set()):
            if neighbor in visited:
                continue
            rate = _nearest_rate(rates, current, neighbor, date)
            if rate is not None:
                visited.add(neighbor)
                queue.append((neighbor, multiplier * rate))
    return None


def _add_image_amounts(
    events: pd.DataFrame,
    image_results: dict[str, ImageExtraction],
    profiles: pd.DataFrame,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    events = events.copy()
    currency_by_user = profiles.set_index("user_id")["home_currency"].to_dict()
    events["image_amount_resolved"] = False
    events["image_anomaly"] = ""
    events["source_currency"] = events["currency"]

    for index, event in events[events["amount"].isna()].iterrows():
        extraction = image_results.get(str(event["event_id"]))
        if extraction is None:
            events.loc[index, "image_anomaly"] = "missing_linked_image"
            continue
        if not extraction.resolved or extraction.amount is None:
            events.loc[index, "image_anomaly"] = extraction.reason or "unresolved_image"
            continue

        source_currency = extraction.currency or event["currency"]
        target_currency = currency_by_user[event["user_id"]]
        cash_date = event["settlement_date"] if pd.notna(event["settlement_date"]) else event["event_date"]
        converted = convert_currency(extraction.amount, source_currency, target_currency, cash_date, rates)

        if converted is None:
            events.loc[index, "image_anomaly"] = f"currency_unresolved:{source_currency}->{target_currency}"
            continue

        events.loc[index, "amount"] = converted
        events.loc[index, "currency"] = target_currency
        events.loc[index, "source_currency"] = source_currency
        events.loc[index, "image_amount_resolved"] = True

        if extraction.amount_paid is not None and extraction.amount_paid == 0:
            events.loc[index, "amount"] = 0.0
            events.loc[index, "status"] = "scheduled"
            events.loc[index, "image_anomaly"] = "billed_unpaid_obligation"

    return events


def _remove_non_cash(events: pd.DataFrame) -> pd.DataFrame:
    return events[
        ~events["event_type"].isin(EXCLUDED_EVENT_TYPES)
        & (events["direction"] != "noncash")
        & events["event_type"].isin(CASH_EVENT_TYPES)
    ].copy()


def _apply_message_policy(
    events: pd.DataFrame,
    message_results: dict[str, MessageExtraction],
) -> pd.DataFrame:
    events = events.copy()
    lookup: dict[str, list[MessageExtraction]] = {}
    for message in message_results.values():
        if message.related_event_id:
            lookup.setdefault(message.related_event_id, []).append(message)

    events["message_anomaly"] = ""
    events["confirmed_by_message"] = True
    for index, event in events.iterrows():
        message_types = {m.message_type for m in lookup.get(str(event["event_id"]), [])}
        if message_types.intersection(UNCONFIRMED_MESSAGE_TYPES) and event["direction"] == "credit":
            events.loc[index, "confirmed_by_message"] = False
            events.loc[index, "message_anomaly"] = ",".join(sorted(message_types))
        if "failed_debit" in message_types:
            events.loc[index, "status"] = "scheduled"
            events.loc[index, "message_anomaly"] = "failed_debit_still_due"
        if "disputed_charge" in message_types:
            events.loc[index, "message_anomaly"] = "disputed_charge_no_reversal"
    return events


def normalize_events(
    events: pd.DataFrame,
    profiles: pd.DataFrame,
    exchange_rates: pd.DataFrame,
    image_results: dict[str, ImageExtraction],
    message_results: dict[str, MessageExtraction],
) -> pd.DataFrame:
    normalized = _add_image_amounts(events, image_results, profiles, exchange_rates)
    normalized = _remove_non_cash(normalized)
    normalized = _apply_message_policy(normalized, message_results)
    normalized["cash_date"] = normalized["settlement_date"].fillna(normalized["event_date"])
    normalized["amount"] = pd.to_numeric(normalized["amount"], errors="coerce")
    normalized = normalized[normalized["amount"].notna()].copy()
    normalized["signed_amount"] = normalized.apply(
        lambda row: float(row["amount"]) if row["direction"] == "credit" else -float(row["amount"]),
        axis=1,
    )
    return normalized


def profile_preferences(profile: pd.Series) -> dict[str, set[str]]:
    return {
        "protected": _split_pipe(profile["expense_categories_to_protect"]),
        "reducible": _split_pipe(profile["expense_categories_user_is_willing_to_reduce"]),
        "stoppable": _split_pipe(profile["expense_categories_user_is_willing_to_stop"]),
        "methods": _split_pipe(profile["payment_methods_user_will_consider"]),
    }
