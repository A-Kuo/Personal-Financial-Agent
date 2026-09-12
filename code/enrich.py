from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import MistralError

from code.config import IMAGE_CACHE_PATH, MEDIA_DIR, MESSAGE_CACHE_PATH
from code.token_tracker import TokenTracker

OCR_MODEL = "mistral-ocr-latest"
# mistral-small-latest and mistral-medium-latest returned a hard 0 req/minute
# entitlement (x-ratelimit-limit-req-minute: 0) on this API key -- confirmed via
# client.models.list() + a live per-model probe. This key is scoped to the
# Ministral 3 family (which is literally what "Mistral 3 key" in .env refers
# to) plus OCR. Verified working with the exact call shape enrich.py uses.
TEXT_MODEL = "ministral-3b-latest"

# USD per official pricing at mistral.ai/pricing/api (Ministral 3 / OCR 4.1 tiers).
# Update these at final submission if provider pricing or the model changes.
OCR_COST_PER_PAGE_USD = 0.004
TEXT_INPUT_COST_PER_TOKEN_USD = 0.1 / 1_000_000
TEXT_OUTPUT_COST_PER_TOKEN_USD = 0.1 / 1_000_000

RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
MAX_RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY_SECONDS = 2.0


def _call_with_retry(func):
    """Call func(), retrying on rate limits / transient server errors with
    exponential backoff (honoring a Retry-After header when the API sends one).
    Mistral's free/low tiers rate-limit aggressively enough that a batch of
    ~230 calls will hit 429s; without this the run dies partway through."""
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return func()
        except MistralError as error:
            if error.status_code == 429 and error.headers.get("x-ratelimit-limit-req-minute") == "0":
                raise RuntimeError(
                    "This model has a permanent 0 requests/minute entitlement on the current "
                    "API key -- retrying will never help. Use a model this key is actually "
                    "provisioned for (check client.models.list() and probe candidates directly)."
                ) from error
            if error.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRY_ATTEMPTS:
                raise
            retry_after = error.headers.get("retry-after")
            delay = float(retry_after) if retry_after else RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay)
    raise RuntimeError("unreachable")


@dataclass(frozen=True)
class ImageExtraction:
    image_id: str
    related_event_id: str
    amount: float | None
    currency: str | None
    resolved: bool
    complete: bool
    amount_paid: float | None
    billed_amount: float | None
    document_type: str
    reason: str | None
    raw_markdown: str


@dataclass(frozen=True)
class MessageExtraction:
    message_id: str
    user_id: str
    related_event_id: str | None
    message_type: str
    confirmed: bool
    amount: float | None
    currency: str | None
    effective_date: str | None
    replacement_date: str | None
    percentage_change: float | None
    notes: str


def _load_json_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json_cache(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_image_path(image_id: str) -> Path:
    exact = list(MEDIA_DIR.glob(f"{image_id}.*"))
    suffixed = list(MEDIA_DIR.glob(f"{image_id}-*"))
    candidates = exact + suffixed
    if not candidates:
        raise FileNotFoundError(f"No media file found for {image_id} in {MEDIA_DIR}")
    return candidates[0]


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime_type};base64,{_encode_image(path)}"


def _client() -> Mistral:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY is missing. Set it in .env or your shell.")
    return Mistral(api_key=api_key)


def _extract_currency(text: str) -> str | None:
    upper = text.upper()
    for code in ("INR", "IDR", "USD", "EUR", "ZAR"):
        if code in upper:
            return code
    if "₹" in text or "RUPEE" in upper or "RS." in upper:
        return "INR"
    if "$" in text:
        return "USD"
    if "€" in text:
        return "EUR"
    return None


def _parse_number(value: str) -> float:
    return float(value.replace(",", "").strip())


def _regex_amount(label: str, text: str) -> float | None:
    pattern = rf"{label}[^0-9]{{0,40}}([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return _parse_number(match.group(1)) if match else None


def _interpret_ocr(image_id: str, related_event_id: str, markdown: str) -> ImageExtraction:
    normalized = " ".join(markdown.split())
    upper = normalized.upper()

    amount_paid = _regex_amount(r"AMOUNT\s+PAID", normalized)
    billed_amount = (
        _regex_amount(r"GRAND\s+TOTAL", normalized)
        or _regex_amount(r"TOTAL\s+BILL\s+AMOUNT", normalized)
        or _regex_amount(r"TOTAL\s+AMOUNT\s+RECEIVED", normalized)
        or _regex_amount(r"NET\s+AMOUNT", normalized)
        or _regex_amount(r"TOTAL", normalized)
    )

    incomplete = (
        "ITEM BILL" in upper
        and "TOTAL PAID" not in upper
        and "GRAND TOTAL" not in upper
        and "TOTAL ORDER" not in upper
    )

    if incomplete:
        return ImageExtraction(
            image_id=image_id,
            related_event_id=related_event_id,
            amount=None,
            currency=_extract_currency(normalized),
            resolved=False,
            complete=False,
            amount_paid=None,
            billed_amount=billed_amount,
            document_type="incomplete_receipt",
            reason="A partial subtotal is visible, but no final total is visible.",
            raw_markdown=markdown,
        )

    if amount_paid is not None:
        return ImageExtraction(
            image_id=image_id,
            related_event_id=related_event_id,
            amount=amount_paid,
            currency=_extract_currency(normalized),
            resolved=True,
            complete=True,
            amount_paid=amount_paid,
            billed_amount=billed_amount,
            document_type="receipt_or_bill",
            reason=None,
            raw_markdown=markdown,
        )

    if billed_amount is None:
        return ImageExtraction(
            image_id=image_id,
            related_event_id=related_event_id,
            amount=None,
            currency=_extract_currency(normalized),
            resolved=False,
            complete=False,
            amount_paid=None,
            billed_amount=None,
            document_type="unknown",
            reason="No reliably labeled final amount was found.",
            raw_markdown=markdown,
        )

    return ImageExtraction(
        image_id=image_id,
        related_event_id=related_event_id,
        amount=billed_amount,
        currency=_extract_currency(normalized),
        resolved=True,
        complete=True,
        amount_paid=None,
        billed_amount=billed_amount,
        document_type="invoice_or_payslip",
        reason=None,
        raw_markdown=markdown,
    )


def extract_images(
    images_df: pd.DataFrame,
    tracker: TokenTracker,
    force_refresh: bool = False,
) -> dict[str, ImageExtraction]:
    cache = _load_json_cache(IMAGE_CACHE_PATH)
    client = _client()
    results: dict[str, ImageExtraction] = {}

    for row in images_df.itertuples(index=False):
        image_id = str(row.image_id)
        event_id = str(row.related_event_id)
        image_path = _find_image_path(image_id)
        cache_key = _file_hash(image_path)

        if not force_refresh and cache_key in cache:
            extraction = ImageExtraction(**cache[cache_key])
            results[event_id] = extraction
            tracker.record(
                request_id=getattr(row, "request_id", None),
                operation="image_ocr",
                provider="mistral",
                model=OCR_MODEL,
                cached=True,
                metadata={"image_id": image_id, "related_event_id": event_id},
            )
            continue

        response = _call_with_retry(
            lambda: client.ocr.process(
                model=OCR_MODEL,
                document={
                    "type": "image_url",
                    "image_url": _image_data_url(image_path),
                },
            )
        )
        markdown = "\n".join(
            getattr(page, "markdown", "") for page in getattr(response, "pages", [])
        )
        extraction = _interpret_ocr(image_id, event_id, markdown)
        cache[cache_key] = extraction.__dict__
        results[event_id] = extraction

        tracker.record(
            request_id=getattr(row, "request_id", None),
            operation="image_ocr",
            provider="mistral",
            model=OCR_MODEL,
            pages_processed=max(len(getattr(response, "pages", [])), 1),
            estimated_cost_usd=OCR_COST_PER_PAGE_USD,
            metadata={
                "image_id": image_id,
                "related_event_id": event_id,
                "resolved": extraction.resolved,
                "complete": extraction.complete,
            },
        )
        time.sleep(1.1)

    _save_json_cache(IMAGE_CACHE_PATH, cache)
    return results


def summarize_image_anomalies(
    images_df: pd.DataFrame, image_results: dict[str, ImageExtraction]
) -> list[str]:
    """Human-readable notes on this run's unresolved or noteworthy image extractions.

    Derived from the actual extraction results, not fixed text, so the note list
    reflects whatever the current run's OCR pass actually found.
    """
    notes: list[str] = []
    for row in images_df.itertuples(index=False):
        event_id = str(row.related_event_id)
        extraction = image_results.get(event_id)
        if extraction is None:
            continue
        request_id = getattr(row, "request_id", "")
        if not extraction.resolved:
            notes.append(f"{event_id} / {request_id}: {extraction.reason or 'unresolved image extraction'}")
        elif extraction.amount_paid == 0:
            notes.append(f"{event_id} / {request_id}: billed amount is due but Amount Paid = 0.00.")
    return notes


_MSG_AMOUNT_RE = re.compile(r"\b(INR|IDR|USD|EUR|ZAR)\s*([\d,]+(?:\.\d{1,2})?)")
_MSG_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MSG_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _msg_first_amount_currency(text: str) -> tuple[float | None, str | None]:
    match = _MSG_AMOUNT_RE.search(text)
    if not match:
        return None, None
    return float(match.group(2).replace(",", "")), match.group(1)


def _msg_first_iso_date(text: str) -> str | None:
    match = _MSG_ISO_DATE_RE.search(text)
    return match.group(1) if match else None


def _msg_first_percent(text: str) -> float | None:
    match = _MSG_PERCENT_RE.search(text)
    return float(match.group(1)) if match else None


# Tier 1: deterministic multilingual template rules (English + Indonesian).
# messages.csv turns out to be ~20 fixed templates translated in parallel across
# both languages, varying only by company name / amount / currency / date --
# exactly the kind of structured-under-the-hood text a hand-written rule can
# classify reliably, without spending a model call on it. Each rule matches
# when ALL phrases in one of its `phrase_sets` appear (case-insensitive) in the
# message; the first matching rule wins, so order runs specific -> generic.
# `wants` says which fields to fill in from the message via regex.
TIER1_MESSAGE_RULES: list[dict] = [
    {"phrase_sets": [["quarterly bonus", "final performance review"], ["bonus kuartalan", "penilaian kinerja"]],
     "type": "unconfirmed_income", "confirmed": False, "wants": set()},
    {"phrase_sets": [["monthly salary has increased to", "the change applies from"], ["gaji bulanan anda naik menjadi", "berlaku mulai"]],
     "type": "salary_change", "confirmed": True, "wants": {"amount", "effective_date"}},
    {"phrase_sets": [["regular salary for the next payroll is", "one-time arrears adjustment"], ["gaji rutin anda untuk penggajian berikutnya adalah", "penyesuaian tunggakan satu kali"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount"}},
    {"phrase_sets": [["gaji rutin untuk penggajian berikutnya sudah dikonfirmasi"]],
     "type": "confirmed_income", "confirmed": True, "wants": set()},
    {"phrase_sets": [["temporary monthly pay"], ["gaji bulanan sementara"]],
     "type": "salary_reduction", "confirmed": True, "wants": {"amount"}},
    {"phrase_sets": [["confirmed salary is now expected on"], ["gaji yang sudah dikonfirmasi kini diperkirakan masuk pada"]],
     "type": "salary_date_change", "confirmed": True, "wants": {"replacement_date"}},
    {"phrase_sets": [["adjustment is due to approved unpaid leave"]],
     "type": "salary_reduction", "confirmed": True, "wants": {"amount"}},
    {"phrase_sets": [["payout is still pending"], ["pembayaran berikutnya", "masih tertunda"]],
     "type": "unconfirmed_income", "confirmed": False, "wants": set()},
    {"phrase_sets": [["confirmed base salary"], ["gaji pokok yang dikonfirmasi"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount"}},
    {"phrase_sets": [["seasonal contract has ended"], ["kontrak musiman saat ini telah berakhir"]],
     "type": "salary_ended", "confirmed": True, "wants": set()},
    {"phrase_sets": [["your employment has ended"], ["hubungan kerja anda telah berakhir"]],
     "type": "salary_ended", "confirmed": True, "wants": set()},
    {"phrase_sets": [["household employment record has ended"], ["sumber pendapatan kerja rumah tangga telah berakhir"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount"}},
    {"phrase_sets": [["resumes on", "childcare"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount", "effective_date"}},
    {"phrase_sets": [["first salary will be", "confirmed credit date is"], ["gaji pertama anda sebesar", "tanggal kredit yang dikonfirmasi"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount", "effective_date"}},
    {"phrase_sets": [["increases monthly rent by"], ["menaikkan biaya sewa bulanan"]],
     "type": "rent_increase", "confirmed": True, "wants": {"percentage_change"}},
    {"phrase_sets": [["transfer between your two accounts"], ["transfer antara dua rekening"]],
     "type": "self_transfer", "confirmed": True, "wants": set()},
    {"phrase_sets": [["refund has been initiated but has not reached"], ["pengembalian dana sudah diproses, tetapi belum masuk"]],
     "type": "refund_pending", "confirmed": False, "wants": set()},
    {"phrase_sets": [["no units have been sold"], ["has not been sold and there has been no cash"], ["belum dijual dan tidak ada transaksi tunai"], ["displayed market value has increased substantially"]],
     "type": "investment_non_cash", "confirmed": False, "wants": set()},
    {"phrase_sets": [["prize claim has been verified and is still in payment processing"], ["klaim hadiah anda sudah diverifikasi dan masih dalam proses pembayaran"]],
     "type": "unconfirmed_income", "confirmed": False, "wants": set()},
    {"phrase_sets": [["prize proceeds have reached your account"]],
     "type": "confirmed_income", "confirmed": True, "wants": set()},
    {"phrase_sets": [["client approved an invoice payment of"], ["klien menyetujui pembayaran faktur sebesar"]],
     "type": "invoice_confirmed", "confirmed": True, "wants": {"amount", "effective_date"}},
    {"phrase_sets": [["previous debit attempt failed"]],
     "type": "failed_debit", "confirmed": True, "wants": set()},
    {"phrase_sets": [["foreign-currency refund is still processing"]],
     "type": "refund_pending", "confirmed": False, "wants": set()},
    {"phrase_sets": [["bill was charged in a foreign currency"], ["tagihan dikenakan dalam mata uang asing"]],
     "type": "other", "confirmed": False, "wants": set()},
    {"phrase_sets": [["pay the release charge"], ["pay the processing charge"], ["bayar biaya pencairan"], ["bayar biaya pemrosesan"]],
     "type": "scam_or_advance_fee", "confirmed": False, "wants": set()},
    {"phrase_sets": [["is scheduled for", "payroll has approved the payment"], ["dijadwalkan pada", "tim payroll sudah menyetujui"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount", "effective_date"}},
    {"phrase_sets": [["first salary from the new employer is"], ["gaji pertama dari perusahaan baru adalah"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount", "effective_date"}},
    {"phrase_sets": [["proceeds from your investment sale have settled"], ["hasil penjualan investasi anda sudah masuk ke rekening tunai"]],
     "type": "investment_sale_settled", "confirmed": True, "wants": set()},
    {"phrase_sets": [["reimbursement for your earlier work expense"], ["penggantian atas biaya kerja"]],
     "type": "other", "confirmed": True, "wants": set()},
    {"phrase_sets": [["reversal has not been posted"], ["dana pembalikannya belum tercatat"]],
     "type": "disputed_charge", "confirmed": False, "wants": set()},
    {"phrase_sets": [["minimum payments due on two separate card accounts"]],
     "type": "other", "confirmed": True, "wants": set()},
    {"phrase_sets": [["is confirmed for", "convert it using the rate"], ["dikonfirmasi untuk", "akan mengonversinya dengan kurs"]],
     "type": "confirmed_income", "confirmed": True, "wants": {"amount", "effective_date"}},
]


def _tier1_classify(message_text: str) -> dict[str, Any] | None:
    """Try to classify a message via fixed template rules. Returns a dict of
    MessageExtraction fields, or None if no rule matches (caller should fall
    back to the Tier 2 model call for these genuinely ambiguous/one-off cases)."""
    lowered = message_text.lower()
    for rule in TIER1_MESSAGE_RULES:
        if not any(all(phrase in lowered for phrase in phrase_set) for phrase_set in rule["phrase_sets"]):
            continue
        wants = rule["wants"]
        amount = currency = None
        if "amount" in wants:
            amount, currency = _msg_first_amount_currency(message_text)
        effective_date = _msg_first_iso_date(message_text) if "effective_date" in wants else None
        replacement_date = _msg_first_iso_date(message_text) if "replacement_date" in wants else None
        percentage_change = _msg_first_percent(message_text) if "percentage_change" in wants else None
        return {
            "message_type": rule["type"],
            "confirmed": rule["confirmed"],
            "amount": amount,
            "currency": currency,
            "effective_date": effective_date,
            "replacement_date": replacement_date,
            "percentage_change": percentage_change,
            "notes": "Classified deterministically by a Tier 1 template rule (no model call).",
        }
    return None


MESSAGE_PROMPT = """
You extract facts from a financial message. The message may be English, Indonesian, or Spanish.
Return JSON only with:
{
  "message_type": "salary_change|salary_date_change|salary_reduction|salary_ended|confirmed_income|unconfirmed_income|invoice_confirmed|refund_pending|refund_settled|self_transfer|failed_debit|disputed_charge|rent_increase|investment_non_cash|investment_sale_settled|scam_or_advance_fee|other",
  "confirmed": true,
  "amount": null,
  "currency": null,
  "effective_date": null,
  "replacement_date": null,
  "percentage_change": null,
  "notes": "short factual summary"
}
Rules: do not infer missing values. Pending payouts, unapproved bonuses, uncredited refunds,
market valuations, and advance-fee prize claims are not confirmed income. Self-transfers are not
income. Failed debits remain obligations. Return JSON only.
"""


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Text model did not return a JSON object.")
    return json.loads(match.group(0))


def extract_messages(
    messages_df: pd.DataFrame,
    tracker: TokenTracker,
    force_refresh: bool = False,
) -> dict[str, MessageExtraction]:
    cache = _load_json_cache(MESSAGE_CACHE_PATH)
    client = _client()
    results: dict[str, MessageExtraction] = {}

    for row in messages_df.itertuples(index=False):
        message_id = str(row.message_id)
        message_text = str(row.message_text)
        related_event_id_raw = getattr(row, "related_event_id", None)
        related_event_id = (
            str(related_event_id_raw)
            if pd.notna(related_event_id_raw) and str(related_event_id_raw).strip()
            else None
        )

        tier1 = _tier1_classify(message_text)
        if tier1 is not None:
            results[message_id] = MessageExtraction(
                message_id=message_id,
                user_id=str(row.user_id),
                related_event_id=related_event_id,
                **tier1,
            )
            continue

        cache_key = hashlib.sha256(message_text.encode("utf-8")).hexdigest()

        if not force_refresh and cache_key in cache:
            extraction = MessageExtraction(**cache[cache_key])
            results[message_id] = extraction
            tracker.record(
                request_id=getattr(row, "request_id", None),
                operation="message_extraction",
                provider="mistral",
                model=TEXT_MODEL,
                cached=True,
                metadata={"message_id": message_id},
            )
            continue

        response = _call_with_retry(
            lambda: client.chat.complete(
                model=TEXT_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": MESSAGE_PROMPT},
                    {"role": "user", "content": message_text},
                ],
                response_format={"type": "json_object"},
            )
        )
        parsed = _parse_json_object(response.choices[0].message.content)
        usage = getattr(response, "usage", None)

        extraction = MessageExtraction(
            message_id=message_id,
            user_id=str(row.user_id),
            related_event_id=related_event_id,
            message_type=str(parsed.get("message_type", "other")),
            confirmed=bool(parsed.get("confirmed", False)),
            amount=parsed.get("amount"),
            currency=parsed.get("currency"),
            effective_date=parsed.get("effective_date"),
            replacement_date=parsed.get("replacement_date"),
            percentage_change=parsed.get("percentage_change"),
            notes=str(parsed.get("notes", "")),
        )
        cache[cache_key] = extraction.__dict__
        results[message_id] = extraction

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        tracker.record(
            request_id=getattr(row, "request_id", None),
            operation="message_extraction",
            provider="mistral",
            model=TEXT_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=(
                input_tokens * TEXT_INPUT_COST_PER_TOKEN_USD
                + output_tokens * TEXT_OUTPUT_COST_PER_TOKEN_USD
            ),
            metadata={"message_id": message_id, "message_type": extraction.message_type},
        )
        time.sleep(1.1)

    _save_json_cache(MESSAGE_CACHE_PATH, cache)
    return results
