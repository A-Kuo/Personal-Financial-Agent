from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from code.config import USAGE_LOG_PATH, USAGE_REPORT_PATH


class TokenTracker:
    def __init__(
        self,
        log_path: Path = USAGE_LOG_PATH,
        report_path: Path = USAGE_REPORT_PATH,
    ) -> None:
        self.log_path = log_path
        self.report_path = report_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()

    def record(
        self,
        *,
        request_id: str | None,
        operation: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        pages_processed: int = 0,
        estimated_cost_usd: float = 0.0,
        cached: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "operation": operation,
            "provider": provider,
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "pages_processed": int(pages_processed or 0),
            "estimated_cost_usd": float(estimated_cost_usd or 0.0),
            "cached": bool(cached),
            "metadata": metadata or {},
        }
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _events(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        with self.log_path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    def write_report(self, total_requests: int, anomalies: list[str] | None = None) -> Path:
        events = self._events()
        by_model: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
            lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "pages_processed": 0,
                "estimated_cost_usd": 0.0,
                "cached_calls": 0,
            }
        )

        for event in events:
            values = by_model[(event["provider"], event["model"], event["operation"])]
            values["calls"] += 1
            values["input_tokens"] += event["input_tokens"]
            values["output_tokens"] += event["output_tokens"]
            values["pages_processed"] += event["pages_processed"]
            values["estimated_cost_usd"] += event["estimated_cost_usd"]
            values["cached_calls"] += int(event["cached"])

        total_input = sum(event["input_tokens"] for event in events)
        total_output = sum(event["output_tokens"] for event in events)
        total_pages = sum(event["pages_processed"] for event in events)
        total_cost = sum(event["estimated_cost_usd"] for event in events)

        lines = [
            "# Model Usage and Cost Report",
            "",
            "This report corresponds to the final pipeline run that generated `output.csv`.",
            "",
            "## Summary",
            "",
            f"- Requests processed: {total_requests}",
            f"- Model calls: {len(events)}",
            f"- Input tokens: {total_input:,}",
            f"- Output tokens: {total_output:,}",
            f"- OCR pages processed: {total_pages}",
            f"- Estimated total cost: ${total_cost:.6f}",
            f"- Average tokens per request: {(total_input + total_output) / max(total_requests, 1):.2f}",
            f"- Average cost per request: ${total_cost / max(total_requests, 1):.6f}",
            "",
            "## Usage by Model",
            "",
            "| Provider | Model | Operation | Calls | Cached calls | Input tokens | Output tokens | OCR pages | Estimated cost (USD) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]

        for (provider, model, operation), values in sorted(by_model.items()):
            lines.append(
                f"| {provider} | {model} | {operation} | {int(values['calls'])} | "
                f"{int(values['cached_calls'])} | {int(values['input_tokens']):,} | "
                f"{int(values['output_tokens']):,} | {int(values['pages_processed'])} | "
                f"${values['estimated_cost_usd']:.6f} |"
            )

        lines.extend(
            [
                "",
                "## Pricing Assumptions",
                "",
                "- OCR calls are costed per processed page using the rate configured in `code/enrich.py`.",
                "- Text-model calls are costed from returned input/output token counts and configured rates.",
                "- Cache hits have zero incremental cost; they are retained for auditability.",
                "",
                "## Anomalies Detected",
                "",
            ]
        )
        if anomalies:
            lines.extend(f"- {anomaly}" for anomaly in anomalies)
        else:
            lines.append("- None detected in this run.")
        lines.append("")

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines), encoding="utf-8")
        return self.report_path
