# Model Usage and Cost Report

This report corresponds to the final pipeline run that generated `output.csv`.

## Summary

- Requests processed: 250
- Model calls: 19
- Input tokens: 884
- Output tokens: 298
- OCR pages processed: 16
- Estimated total cost: $0.064118
- Average tokens per request: 4.73
- Average cost per request: $0.000256

## Usage by Model

| Provider | Model | Operation | Calls | Cached calls | Input tokens | Output tokens | OCR pages | Estimated cost (USD) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mistral | ministral-3b-latest | message_extraction | 3 | 0 | 884 | 298 | 0 | $0.000118 |
| mistral | mistral-ocr-latest | image_ocr | 16 | 0 | 0 | 0 | 16 | $0.064000 |

## Pricing Assumptions

- OCR calls are costed per processed page using the rate configured in `code/enrich.py`.
- Text-model calls are costed from returned input/output token counts and configured rates.
- Cache hits have zero incremental cost; they are retained for auditability.

## Anomalies Detected

- event_1700 / request_19: No reliably labeled final amount was found.
- event_6859 / request_73: billed amount is due but Amount Paid = 0.00.
