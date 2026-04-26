"""I/O helpers: checksums, manifest generation, parquet writing."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path, partition_cols: list[str] | None = None) -> None:
    """Write a DataFrame to Parquet, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_cols:
        df.to_parquet(path.parent, partition_cols=partition_cols, index=False)
    else:
        df.to_parquet(path, index=False)


def build_manifest(zone_dir: Path) -> dict:
    """
    Walk a datalake zone directory and build a manifest.json describing
    each Parquet file: name, row count, schema, timestamp, checksum.
    """
    manifest = {
        "zone": zone_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [],
    }

    for p in sorted(zone_dir.rglob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
            entry = {
                "file": str(p.relative_to(zone_dir)),
                "rows": len(df),
                "schema": schema,
                "sha256": sha256_file(p),
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            manifest["files"].append(entry)
        except Exception:
            pass  # Non-parquet files are skipped gracefully

    return manifest


def write_manifest(zone_dir: Path) -> None:
    """Generate and write manifest.json for a datalake zone."""
    manifest = build_manifest(zone_dir)
    out = zone_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
