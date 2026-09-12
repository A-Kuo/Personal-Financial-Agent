# Model Usage and Cost Report

This report corresponds to the final pipeline run that generated `output.csv`.

## Summary

- Requests processed: 250
- Model calls: 19
- Input tokens: 0
- Output tokens: 0
- OCR pages processed: 0
- Estimated total cost: $0.000000
- Average tokens per request: 0.00
- Average cost per request: $0.000000

## Usage by Model

| Provider | Model | Operation | Calls | Cached calls | Input tokens | Output tokens | OCR pages | Estimated cost (USD) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mistral | ministral-3b-latest | message_extraction | 3 | 3 | 0 | 0 | 0 | $0.000000 |
| mistral | mistral-ocr-latest | image_ocr | 16 | 16 | 0 | 0 | 0 | $0.000000 |

## Pricing Assumptions

- OCR calls are costed per processed page using the rate configured in `code/enrich.py`.
- Text-model calls are costed from returned input/output token counts and configured rates.
- Cache hits have zero incremental cost; they are retained for auditability.

## Anomalies Detected

- event_1700 / request_19: No reliably labeled final amount was found.
- event_6859 / request_73: billed amount is due but Amount Paid = 0.00.
