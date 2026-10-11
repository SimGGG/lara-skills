"""Versioned data output, integrity helpers and the transient-workspace lifecycle."""

import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ._version import __version__

PACKAGE_NAME = "lara-skills"
TRACKED_PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "duckdb", "torch", "sentence-transformers", "openpyxl"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_parquet(frame, path):
    Path(path).write_bytes(frame.to_parquet(index=False))


def package_versions():
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def code_hashes():
    """Hashes of the installed package modules; replaces project-root script hashes."""
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def environment():
    return {"python": sys.version, "platform": platform.platform(), "packages": package_versions()}


def read_table(path, sheet_name=None):
    """Read a CSV, Excel or Parquet table with every column as text (missing cells stay missing)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0, dtype=str)
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.apply(lambda column: column.map(lambda v: None if pd.isna(v) else str(v)))
    raise ValueError(f"Unsupported table format: {path.name} (use .csv, .xlsx or .parquet)")


@contextmanager
def output_run(root, destination, purpose, config):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(root) or destination == root:
        raise ValueError("Output must be a versioned subdirectory of the workspace root")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + purpose
    run_dir = root / "artifacts/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "purpose": purpose, "status": "running", "config": config,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": destination.relative_to(root).as_posix(),
                "package": {"name": PACKAGE_NAME, "version": __version__},
                "command": list(sys.argv), "environment": environment(), "code_hashes": code_hashes()}
    write_json(run_dir / "manifest.json", manifest)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=destination.name + ".partial_", dir=destination.parent) as tmp:
            staging = Path(tmp)
            yield staging, manifest
            manifest.update({"status": "completed", "finished_at": datetime.now(timezone.utc).isoformat(),
                             "files": {p.relative_to(staging).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                                       for p in sorted(staging.rglob("*")) if p.is_file()}})
            write_json(staging / "manifest.json", manifest)
            staging.rename(destination)
        write_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"completed": purpose, "run_id": run_id, "output": str(destination)}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        manifest.update({"status": "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                         "error": repr(exc), "finished_at": datetime.now(timezone.utc).isoformat()})
        write_json(run_dir / "manifest.json", manifest)
        raise
