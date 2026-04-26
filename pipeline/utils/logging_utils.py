"""Structured logging utilities for the Clovertex pipeline."""

import json
import sys
from datetime import datetime, timezone


def log_dataset_stats(
    dataset: str,
    rows_in: int,
    rows_out: int,
    issues_found: dict,
) -> None:
    """Emit a structured JSON log line to stdout for each processed dataset."""
    record = {
        "dataset": dataset,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "issues_found": issues_found,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(record), flush=True)


def log_info(message: str) -> None:
    """Print a human-readable info line to stderr so it doesn't pollute JSON stdout."""
    print(f"[INFO]  {message}", file=sys.stderr, flush=True)


def log_warn(message: str) -> None:
    """Print a warning to stderr."""
    print(f"[WARN]  {message}", file=sys.stderr, flush=True)


def log_error(message: str) -> None:
    """Print an error to stderr."""
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
