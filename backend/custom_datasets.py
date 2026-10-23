"""CNC Sentinel — Custom dataset import (Phase 1: upload, preview, mapping, validation).

Scope of this module:
- Stores uploaded CSV files under ``data/custom/<dataset_id>/`` (Git-ignored), fully separate
  from the NASA dataset (data/raw/mill.mat) and the production artifacts (artifacts/).
- Parses CSV cells strictly as text data; nothing is evaluated or executed.
- Profiles the upload, records a user-confirmed column mapping, and writes a validation and
  model-compatibility report.

It never loads or calls the production model: compatibility is judged against the documented
feature contract (ml/feature_config.py, ml/prepare_data.py constants) only. Predictions for
uploaded data are always reported as unavailable in this phase.
"""

from __future__ import annotations

import codecs
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ml.feature_config import COMBINED_FEATURES
from ml.prepare_data import SAMPLING_RATE_HZ, SENSOR_CHANNELS, SPECTRAL_BANDS

REPO_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = Path(os.environ.get("CNC_SENTINEL_DATASET_DIR", REPO_ROOT / "data" / "custom"))

# ---------------------------------------------------------------------------- limits
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_ROWS = 200_000
MAX_COLUMNS = 500
MAX_COLUMN_NAME_LEN = 128
MAX_FIELD_CHARS = 100_000
PREVIEW_ROWS = 20
MAX_PREVIEW_LIMIT = 200
MAX_CELL_CHARS = 500
SNIFF_BYTES = 64 * 1024
DELIMITERS = [",", ";", "\t", "|"]

BUILTIN_ID = "nasa_mill"
ID_RE = re.compile(r"^ds_[0-9a-f]{32}$")

# Tokens treated as an explicit missing value; non-finite tokens are counted separately.
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}
NONFINITE_TOKENS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
INT_RE = re.compile(r"^[+-]?\d+$")
DECIMAL_COMMA_RE = re.compile(r"^[+-]?\d+,\d+$")
FORMULA_PREFIXES = ("=", "+", "-", "@")

# ---------------------------------------------------------------------------- NASA feature contract (documented, not learned)
NASA_SAMPLES_PER_CUT = 9000  # ml/prepare_data.evaluate_run_quality flags any other length VARIABLE_LENGTH
NASA_CHANNELS = [prefix for _, prefix in SENSOR_CHANNELS]
NASA_FEED_LEVELS = (0.25, 0.5)        # mm/rev, the only levels in artifacts/cut_features.csv
NASA_DOC_LEVELS = (0.75, 1.5)         # mm
NASA_MATERIAL_CODES = (1, 2)          # 1 = cast iron, 2 = steel (mill.mat coding)
NASA_CUTTING_TIME_RANGE = (0.0, 105.0)  # cumulative minutes
OPERATING_FEATURES = {"material": "material_id", "feed_rate": "feed_rate_mm_rev",
                      "depth_of_cut": "depth_of_cut_mm", "cutting_time": "cutting_time_min"}

LAYOUTS = ("per_cut_features", "raw_samples")
SINGLE_ROLES = ("tool_id", "cut_index", "timestamp", "measured_wear", "target", "material", "feed_rate", "depth_of_cut", "cutting_time")
ROLES = SINGLE_ROLES + ("sensor", "feature", "ignore")
ROLE_UNITS: Dict[str, Tuple[str, ...]] = {
    "measured_wear": ("mm", "um"),
    "target": ("mm", "um"),
    "timestamp": ("s", "ms", "datetime"),
    "cutting_time": ("min", "s"),
    "feed_rate": ("mm_per_rev", "unknown"),
    "depth_of_cut": ("mm", "unknown"),
    "sensor": ("V", "mV", "A", "g", "m/s2", "N", "unknown"),
}
SENSOR_CHANNEL_CHOICES = tuple(NASA_CHANNELS) + ("other",)
NUMERIC_ROLES = {"measured_wear", "target", "material", "feed_rate", "depth_of_cut", "cutting_time", "sensor", "feature"}

_lock = threading.Lock()


class DatasetError(Exception):
    """A request that cannot be fulfilled; carries an HTTP status and a stable error code."""

    def __init__(self, status: int, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details or {}

    def to_detail(self) -> Dict[str, Any]:
        return {"error_code": self.code, "message": self.message, "details": self.details}


# ============================================================================ storage helpers
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dataset_dir(dataset_id: str) -> Path:
    """Resolve a dataset directory; any malformed or unknown id is reported as not found."""
    if dataset_id == BUILTIN_ID:
        raise DatasetError(404, "BUILTIN_DATASET_NOT_IMPORTED",
                           "The NASA dataset is built in and read-only; it has no import record.")
    if not isinstance(dataset_id, str) or not ID_RE.match(dataset_id):
        raise DatasetError(404, "DATASET_NOT_FOUND", "No imported dataset has this id.")
    root = STORAGE_ROOT.resolve()
    path = (root / dataset_id).resolve()
    if path.parent != root or not (path / "manifest.json").is_file():
        raise DatasetError(404, "DATASET_NOT_FOUND", "No imported dataset has this id.")
    return path


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def _manifest(dataset_id: str) -> Tuple[Path, Dict[str, Any]]:
    d = _dataset_dir(dataset_id)
    m = _read_json(d / "manifest.json")
    if m.get("dataset_id") != dataset_id:  # defence in depth: never serve another dataset's record
        raise DatasetError(404, "DATASET_NOT_FOUND", "No imported dataset has this id.")
    return d, m


def _clean_name(name: Optional[str], fallback: str) -> str:
    text = "".join(ch for ch in (name or "") if ch.isprintable()).strip()[:80]
    return text or fallback


# ============================================================================ cell classification
def classify(token: str) -> Tuple[str, Optional[float]]:
    """Classify one CSV cell: ('missing'|'nonfinite'|'number'|'text', value)."""
    t = token.strip()
    low = t.lower()
    if low in MISSING_TOKENS:
        return "missing", None
    if low in NONFINITE_TOKENS:
        return "nonfinite", None
    if NUMBER_RE.match(t):
        v = float(t)
        return ("number", v) if math.isfinite(v) else ("nonfinite", None)
    return "text", None


def _is_datetime(token: str) -> bool:
    t = token.strip()
    if len(t) < 8 or not any(c in t for c in "-:/T"):
        return False
    try:
        datetime.fromisoformat(t.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _parse_datetime_seconds(token: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(token.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ============================================================================ upload
class IncomingUpload:
    """Streams an upload to a private temp file with size, binary and UTF-8 checks applied per chunk."""

    def __init__(self, filename: str, name: Optional[str]):
        if not isinstance(filename, str) or not filename.lower().endswith(".csv"):
            raise DatasetError(415, "UNSUPPORTED_FILE_TYPE", "Only .csv files can be imported.",
                               {"filename": str(filename)[:200]})
        self.filename = os.path.basename(filename.replace("\\", "/"))[:200]
        self.name = _clean_name(name, self.filename[:-4] or "Imported dataset")
        incoming = STORAGE_ROOT / ".incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        self.path = incoming / (uuid.uuid4().hex + ".part")
        self._f = open(self.path, "wb")
        self.size = 0
        self._sha = hashlib.sha256()
        self._dec = codecs.getincrementaldecoder("utf-8")("strict")
        self._first = True
        self.bom = False

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.size += len(chunk)
        if self.size > MAX_UPLOAD_BYTES:
            raise DatasetError(413, "FILE_TOO_LARGE", f"The file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB import limit.",
                               {"limit_bytes": MAX_UPLOAD_BYTES})
        if self._first:
            self._first = False
            self.bom = chunk.startswith(codecs.BOM_UTF8)
        if b"\x00" in chunk:
            raise DatasetError(415, "BINARY_CONTENT", "The file contains NUL bytes; it is not a text CSV file.")
        try:
            self._dec.decode(chunk, final=False)
        except UnicodeDecodeError:
            raise DatasetError(415, "UNSUPPORTED_ENCODING", "The file is not valid UTF-8 text. Re-save it as UTF-8 CSV.")
        self._sha.update(chunk)
        self._f.write(chunk)

    def abort(self) -> None:
        try:
            self._f.close()
        finally:
            self.path.unlink(missing_ok=True)

    def finish(self) -> Dict[str, Any]:
        """Close the stream, parse and profile it, and commit it as a new dataset."""
        try:
            self._f.close()
            try:
                self._dec.decode(b"", final=True)
            except UnicodeDecodeError:
                raise DatasetError(415, "UNSUPPORTED_ENCODING", "The file ends inside a multi-byte character; it is not valid UTF-8.")
            if self.size == 0:
                raise DatasetError(422, "EMPTY_FILE", "The uploaded file is empty.")
            encoding = "utf-8-sig" if self.bom else "utf-8"
            delimiter = _sniff_delimiter(self.path, encoding)
            profile = _profile(self.path, encoding, delimiter)
        except Exception:
            self.abort()
            raise

        dataset_id = "ds_" + uuid.uuid4().hex
        d = STORAGE_ROOT / dataset_id
        d.mkdir(parents=True, exist_ok=False)
        original = d / "original.csv"
        os.replace(self.path, original)
        os.chmod(original, stat.S_IREAD)  # the original upload is never modified
        manifest = {
            "dataset_id": dataset_id, "source": "CUSTOM_UPLOAD", "name": self.name, "original_filename": self.filename,
            "uploaded_at": _now(), "size_bytes": self.size, "sha256": self._sha.hexdigest(), "encoding": encoding,
            "delimiter": delimiter, "n_rows": profile["n_rows"], "n_columns": profile["n_columns"],
            "status": "UPLOADED", "mapping_version": 0, "validated_mapping_version": None,
            "compatibility_verdict": None, "predictions_available": False,
        }
        _write_json(d / "profile.json", profile)
        _write_json(d / "manifest.json", manifest)
        return get_detail(dataset_id)


def _sniff_delimiter(path: Path, encoding: str) -> str:
    with open(path, "r", encoding=encoding, newline="") as f:
        sample = f.read(SNIFF_BYTES)
    lines = sample.splitlines()
    if len(sample) == SNIFF_BYTES and len(lines) > 1:
        lines = lines[:-1]  # the last line may be cut mid-row
    lines = [ln for ln in lines if ln.strip()][:50]
    if not lines:
        raise DatasetError(422, "EMPTY_FILE", "The file contains no text rows.")
    best, best_n = None, 1
    for d in DELIMITERS:
        try:
            rows = list(csv.reader(lines, delimiter=d, strict=True))
        except csv.Error:
            continue
        counts = {len(r) for r in rows}
        n = len(rows[0])
        if n > best_n and counts == {n}:
            best, best_n = d, n
    if best is None:
        for d in DELIMITERS:  # header splits but some sampled rows are ragged: still report the structure error
            try:
                if len(next(csv.reader([lines[0]], delimiter=d))) > 1:
                    return d
            except csv.Error:
                continue
        raise DatasetError(422, "UNSUPPORTED_STRUCTURE",
                           "Could not detect a column delimiter. Use comma, semicolon, tab or pipe with at least two columns.")
    return best


def _iter_rows(path: Path, encoding: str, delimiter: str) -> Iterator[Tuple[int, List[str]]]:
    """Yield (line_number, row) for non-blank rows; csv errors are reported with their line number."""
    old = csv.field_size_limit()
    csv.field_size_limit(MAX_FIELD_CHARS)
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter, strict=True)
            while True:
                try:
                    row = next(reader)
                except StopIteration:
                    return
                except csv.Error as exc:
                    raise DatasetError(422, "MALFORMED_CSV", f"CSV parsing failed near line {reader.line_num}: {exc}.",
                                       {"line": reader.line_num})
                if not row or all(not c.strip() for c in row) and len(row) <= 1:
                    continue
                yield reader.line_num, row
    finally:
        csv.field_size_limit(old)


def _profile(path: Path, encoding: str, delimiter: str) -> Dict[str, Any]:
    with contextlib.closing(_iter_rows(path, encoding, delimiter)) as rows:
        return _profile_rows(rows)


def _profile_rows(rows: Iterator[Tuple[int, List[str]]]) -> Dict[str, Any]:
    try:
        _, header = next(rows)
    except StopIteration:
        raise DatasetError(422, "EMPTY_FILE", "The file contains no rows.")
    header = [h.strip() for h in header]
    _check_header(header)
    n_cols = len(header)

    cols = [{"missing": 0, "nonfinite": 0, "number": 0, "integer": 0, "text": 0, "boolean": 0, "datetime": 0,
             "decimal_comma": 0, "formula_like": 0, "min": None, "max": None, "distinct": set(), "distinct_overflow": False,
             "examples": []} for _ in header]
    seen: set = set()
    n_rows = duplicates = 0
    ragged: List[int] = []
    preview: List[List[str]] = []
    for line, row in rows:
        if len(row) != n_cols:
            if len(ragged) < 10:
                ragged.append(line)
            else:
                break
            continue
        n_rows += 1
        if n_rows > MAX_ROWS:
            raise DatasetError(413, "ROW_LIMIT_EXCEEDED", f"The file has more than {MAX_ROWS:,} data rows.", {"limit_rows": MAX_ROWS})
        key = hashlib.blake2b("\x1f".join(row).encode("utf-8"), digest_size=12).digest()
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
        if len(preview) < PREVIEW_ROWS:
            preview.append([_cell(c) for c in row])
        for c, tok in zip(cols, row):
            kind, v = classify(tok)
            c[kind] += 1
            if kind == "number":
                if INT_RE.match(tok.strip()):
                    c["integer"] += 1
                c["min"] = v if c["min"] is None else min(c["min"], v)
                c["max"] = v if c["max"] is None else max(c["max"], v)
            elif kind == "text":
                s = tok.strip()
                if s.lower() in ("true", "false"):
                    c["boolean"] += 1
                elif _is_datetime(s):
                    c["datetime"] += 1
                elif DECIMAL_COMMA_RE.match(s):
                    c["decimal_comma"] += 1
                if s.startswith(FORMULA_PREFIXES) and not NUMBER_RE.match(s):
                    c["formula_like"] += 1
            if kind != "missing":
                if len(c["examples"]) < 3:
                    c["examples"].append(_cell(tok.strip()))
                if not c["distinct_overflow"]:
                    c["distinct"].add(tok.strip())
                    if len(c["distinct"]) > 1000:
                        c["distinct_overflow"], c["distinct"] = True, set()
    if ragged:
        raise DatasetError(422, "MALFORMED_ROWS",
                           f"Some rows do not have {n_cols} fields (first at line {ragged[0]}). Every row must match the header.",
                           {"expected_fields": n_cols, "lines": ragged})
    if n_rows == 0:
        raise DatasetError(422, "NO_DATA_ROWS", "The file has a header row but no data rows.")

    columns, warnings = [], []
    for name, c in zip(header, cols):
        present = n_rows - c["missing"]
        if present == 0:
            dtype = "empty"
        elif c["text"] == 0:
            dtype = "integer" if c["integer"] == c["number"] and c["nonfinite"] == 0 else "float"
        elif c["boolean"] == present:
            dtype = "boolean"
        elif c["datetime"] == present:
            dtype = "datetime"
        else:
            dtype = "text"
        distinct = None if c["distinct_overflow"] else len(c["distinct"])
        columns.append({"name": name, "inferred_type": dtype, "non_missing": present, "missing": c["missing"],
                        "non_finite": c["nonfinite"], "non_numeric": c["text"],
                        "numeric_values": c["number"], "distinct": distinct, "min": c["min"], "max": c["max"], "examples": c["examples"]})
        if dtype == "empty":
            warnings.append(_warn("EMPTY_COLUMN", f"Column '{name}' has no values.", name))
        elif c["missing"]:
            warnings.append(_warn("MISSING_VALUES", f"Column '{name}' has {c['missing']} missing value(s); they stay missing (never zero).", name))
        if c["nonfinite"]:
            warnings.append(_warn("NON_FINITE_VALUES", f"Column '{name}' has {c['nonfinite']} non-finite value(s) (inf); they are not usable numbers.", name))
        if dtype == "text" and c["number"] and c["number"] >= 0.5 * present:
            warnings.append(_warn("MIXED_TYPES", f"Column '{name}' is mostly numeric but has {c['text']} non-numeric value(s).", name))
        if c["decimal_comma"] and c["decimal_comma"] >= 0.5 * present:
            warnings.append(_warn("DECIMAL_COMMA_SUSPECTED", f"Column '{name}' looks like decimal commas (e.g. 0,25); values are kept as text, not converted.", name))
        if distinct == 1 and n_rows > 1:
            warnings.append(_warn("CONSTANT_COLUMN", f"Column '{name}' has a single distinct value.", name))
        if c["formula_like"]:
            warnings.append(_warn("FORMULA_LIKE_TEXT", f"Column '{name}' has {c['formula_like']} cell(s) starting with =, +, - or @; they are stored and shown as plain text only.", name, "info"))
    if duplicates:
        warnings.append(_warn("DUPLICATE_ROWS", f"{duplicates} row(s) exactly duplicate an earlier row.", None))
    return {"n_rows": n_rows, "n_columns": n_cols, "columns": columns, "duplicate_rows": duplicates,
            "missing_cells": sum(c["missing"] for c in cols), "non_finite_cells": sum(c["nonfinite"] for c in cols),
            "preview_rows": preview, "warnings": warnings,
            "missing_value_tokens": sorted(MISSING_TOKENS), "non_finite_tokens": sorted(NONFINITE_TOKENS)}


def _check_header(header: List[str]) -> None:
    if len(header) < 2:
        raise DatasetError(422, "UNSUPPORTED_STRUCTURE", "At least two columns are required.")
    if len(header) > MAX_COLUMNS:
        raise DatasetError(413, "TOO_MANY_COLUMNS", f"The file has {len(header)} columns; the limit is {MAX_COLUMNS}.",
                           {"limit_columns": MAX_COLUMNS})
    empty = [i + 1 for i, h in enumerate(header) if not h]
    if empty:
        raise DatasetError(422, "MISSING_HEADER", f"Column(s) {empty} have no header name. Every column needs a unique name.",
                           {"positions": empty})
    long_names = [h[:40] for h in header if len(h) > MAX_COLUMN_NAME_LEN]
    if long_names:
        raise DatasetError(422, "HEADER_TOO_LONG", f"Column names are limited to {MAX_COLUMN_NAME_LEN} characters.", {"columns": long_names})
    seen: Dict[str, str] = {}
    dups = []
    for h in header:
        k = h.casefold()
        if k in seen:
            dups.append(h if h == seen[k] else f"{seen[k]} / {h}")
        seen.setdefault(k, h)
    if dups:
        raise DatasetError(422, "DUPLICATE_COLUMN_NAMES", "Column names must be unique (case-insensitive): " + ", ".join(dups[:10]) + ".",
                           {"duplicates": dups[:50]})
    if all(classify(h)[0] in ("number", "nonfinite") for h in header):
        raise DatasetError(422, "MISSING_HEADER", "The first row looks like data, not column names. Add a header row.")


def _cell(v: str) -> str:
    return v if len(v) <= MAX_CELL_CHARS else v[:MAX_CELL_CHARS] + "…"


def _warn(code: str, message: str, column: Optional[str], severity: str = "warning") -> Dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "column": column}


# ============================================================================ read API
def list_datasets() -> Dict[str, Any]:
    items = []
    root = STORAGE_ROOT
    if root.is_dir():
        for p in sorted(root.iterdir()):
            if p.is_dir() and ID_RE.match(p.name) and (p / "manifest.json").is_file():
                try:
                    items.append(_read_json(p / "manifest.json"))
                except (OSError, ValueError):
                    continue
    items.sort(key=lambda m: m.get("uploaded_at", ""), reverse=True)
    return {"builtin": builtin_summary(), "datasets": items}


def builtin_summary() -> Dict[str, Any]:
    from backend import services  # read-only facts about the built-in feature table; never the model

    return {"dataset_id": BUILTIN_ID, "source": "NASA_RECORDED", "name": "NASA Milling (built-in)", "read_only": True,
            "original_filename": "data/raw/mill.mat", "feature_table_present": services.CUT_FEATURES_PATH.exists(),
            "description": "Recorded NASA milling runs used to train and verify the production model."}


def get_detail(dataset_id: str) -> Dict[str, Any]:
    d, m = _manifest(dataset_id)
    profile = _read_json(d / "profile.json")
    mapping = _read_json(d / "mapping.json") if (d / "mapping.json").is_file() else None
    validation = _read_json(d / "validation.json") if (d / "validation.json").is_file() else None
    return {"manifest": m, "profile": {k: v for k, v in profile.items() if k != "preview_rows"}, "mapping": mapping,
            "validation_summary": _validation_summary(validation, m)}


def _validation_summary(v: Optional[Dict[str, Any]], m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not v:
        return None
    return {"validated_at": v["validated_at"], "mapping_version": v["mapping_version"],
            "stale": v["mapping_version"] != m.get("mapping_version"), "overall_status": v["overall_status"],
            "compatibility_verdict": v["model_compatibility"]["verdict"], "predictions_available": False}


def get_preview(dataset_id: str, offset: int, limit: int) -> Dict[str, Any]:
    d, m = _manifest(dataset_id)
    if offset < 0 or not (1 <= limit <= MAX_PREVIEW_LIMIT):
        raise DatasetError(422, "INVALID_PREVIEW_RANGE", f"offset must be >= 0 and limit between 1 and {MAX_PREVIEW_LIMIT}.")
    rows: List[List[str]] = []
    with contextlib.closing(_iter_rows(d / "original.csv", m["encoding"], m["delimiter"])) as it:
        _, header = next(it)
        for i, (_, row) in enumerate(it):
            if i < offset:
                continue
            if len(rows) >= limit:
                break
            rows.append([_cell(c) for c in row])
    return {"dataset_id": dataset_id, "columns": [h.strip() for h in header], "offset": offset, "limit": limit,
            "total_rows": m["n_rows"], "rows": rows}


def get_validation(dataset_id: str) -> Dict[str, Any]:
    d, m = _manifest(dataset_id)
    if not (d / "validation.json").is_file():
        raise DatasetError(404, "VALIDATION_NOT_RUN", "Validation has not been run for this dataset yet.")
    v = _read_json(d / "validation.json")
    v["stale"] = v["mapping_version"] != m.get("mapping_version")
    return v


def delete_dataset(dataset_id: str) -> None:
    if dataset_id == BUILTIN_ID:
        raise DatasetError(403, "BUILTIN_DATASET_READ_ONLY", "The built-in NASA dataset cannot be removed.")
    d, _ = _manifest(dataset_id)
    root = STORAGE_ROOT.resolve()
    if d.parent != root or not ID_RE.match(d.name):
        raise DatasetError(404, "DATASET_NOT_FOUND", "No imported dataset has this id.")

    def _unlock(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    with _lock:
        shutil.rmtree(d, onerror=_unlock)


# ============================================================================ mapping
_NAME_HINTS: List[Tuple[str, Tuple[str, ...]]] = [
    ("tool_id", ("toolid", "tool", "case", "toolname", "insert", "insertid", "cutterid")),
    ("cut_index", ("cutindex", "cut", "run", "runid", "pass", "passno", "cutno", "cutnumber", "runnumber", "run_num")),
    ("timestamp", ("timestamp", "ts", "datetime", "sampletime", "times", "timesec", "t")),
    ("measured_wear", ("measuredwearvb", "vb", "flankwear", "flankwearvb", "wear", "wearvb", "toolwear", "vbmm", "vbum", "wearmm", "wearum", "flankwearmm", "flankwearum", "toolwearmm", "toolwearum", "target", "targetwear", "targetwearmm", "targetwearvb")),
    ("material", ("material", "materialid")),
    ("feed_rate", ("feed", "feedrate", "feedratemmrev", "feedmmrev")),
    ("depth_of_cut", ("doc", "depthofcut", "depthofcutmm", "ap")),
    ("cutting_time", ("cuttingtime", "cuttingtimemin", "time")),
]
_CHANNEL_HINTS = {"smcac": "smc_ac", "smcdc": "smc_dc", "vibtable": "vib_table", "vibspindle": "vib_spindle",
                  "aetable": "ae_table", "aespindle": "ae_spindle"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def suggest_mapping(dataset_id: str) -> Dict[str, Any]:
    """Name-based suggestions only. Ambiguous roles are left unassigned; units are never guessed."""
    d, m = _manifest(dataset_id)
    profile = _read_json(d / "profile.json")
    names = [c["name"] for c in profile["columns"]]
    features = set(COMBINED_FEATURES)
    candidates: Dict[str, List[str]] = {r: [] for r in SINGLE_ROLES}
    columns: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    for name in names:
        n = _norm(name)
        if name in features:
            columns[name] = {"role": "feature", "feature": name}
            continue
        if n in _CHANNEL_HINTS:
            columns[name] = {"role": "sensor", "channel": _CHANNEL_HINTS[n], "unit": "unknown"}
            continue
        for role, hints in _NAME_HINTS:
            if n in hints:
                candidates[role].append(name)
                break
    ambiguities = []
    for role, cols in candidates.items():
        if len(cols) == 1:
            entry: Dict[str, Any] = {"role": role}
            if role in ("measured_wear", "target"):
                n = _norm(cols[0])
                if n.endswith("mm"):
                    entry["unit"] = "mm"; notes.append(f"Wear unit 'mm' suggested from the column name '{cols[0]}'; confirm it.")
                elif n.endswith("um"):
                    entry["unit"] = "um"; notes.append(f"Wear unit 'um' suggested from the column name '{cols[0]}'; confirm it.")
            columns[cols[0]] = entry
        elif len(cols) > 1:
            ambiguities.append({"role": role, "candidates": cols,
                                "message": f"Several columns could be {role}: {', '.join(cols)}. Choose one."})
    n_features = sum(1 for v in columns.values() if v["role"] == "feature")
    n_sensors = sum(1 for v in columns.values() if v["role"] == "sensor")
    layout = "per_cut_features" if n_features >= n_sensors and n_features > 0 else "raw_samples" if n_sensors else None
    tool = next((c for c, v in columns.items() if v["role"] == "tool_id"), None)
    cut = next((c for c, v in columns.items() if v["role"] == "cut_index"), None)
    if tool and cut:
        per_cut = _median_rows_per_cut(d, m, names.index(tool), names.index(cut))
        layout = "raw_samples" if per_cut >= 2 else "per_cut_features"
        notes.append(f"Layout suggested from the data: a typical (tool, cut) pair has {per_cut:g} distinct row(s)"
                     + (" (several samples per cut)." if per_cut >= 2 else " (one row per cut); exact duplicate rows are not counted."))
    if layout == "per_cut_features":
        # Remaining numeric columns become sensor/operating features, except names that look like wear,
        # a target or a prediction: those are never silently used as model inputs.
        types = {c["name"]: c["inferred_type"] for c in profile["columns"]}
        ambiguous_cols = {c for a in ambiguities for c in a["candidates"]}
        auto = []
        for name in names:
            if name in columns or name in ambiguous_cols or types.get(name) not in ("integer", "float"):
                continue
            if _WEAR_LIKE.search(_norm(name)):
                notes.append(f"'{name}' looks wear- or target-related, so it is not used as an input feature. Map it manually if it is a sensor reading.")
                continue
            columns[name] = {"role": "feature", "feature": "other"}
            auto.append(name)
        if auto:
            notes.append(f"{len(auto)} numeric column(s) used as sensor/operating features: {', '.join(auto[:8])}{'…' if len(auto) > 8 else ''}.")
    return {"dataset_id": dataset_id, "layout": layout, "columns": columns, "sampling_rate_hz": None,
            "ambiguities": ambiguities, "notes": notes}


_WEAR_LIKE = re.compile(r"wear|vb|target|pred|rul|label")
DOCUMENTED_FORMAT = ("One row per cut: tool_id, cut_number, wear_mm (optional, measured flank wear in mm), "
                     "then any numeric sensor or operating columns (e.g. spindle_speed, feed_rate, cutting_force, vibration, temperature).")


def auto_map(dataset_id: str) -> Dict[str, Any]:
    """Quick setup for the documented per-cut format: save the suggested mapping, or explain how to fix the CSV."""
    s = suggest_mapping(dataset_id)
    cols = s["columns"]
    roles = {spec["role"]: c for c, spec in cols.items()}
    amb_roles = {a["role"] for a in s["ambiguities"]}
    problems = []
    for a in s["ambiguities"]:
        problems.append({"code": "AMBIGUOUS_COLUMNS", "message": a["message"],
                         "fix": f"Keep exactly one {a['role']} column (rename or remove: {', '.join(a['candidates'])})."})
    if "tool_id" not in roles and "tool_id" not in amb_roles:
        problems.append({"code": "MISSING_TOOL_ID", "message": "No tool identifier column was recognised.",
                         "fix": "Add a column named tool_id that identifies each tool or insert."})
    if "cut_index" not in roles and "cut_index" not in amb_roles:
        problems.append({"code": "MISSING_CUT_NUMBER", "message": "No cut number column was recognised.",
                         "fix": "Add a column named cut_number with the cut or pass number (1, 2, 3, …)."})
    if s["layout"] == "raw_samples" and "tool_id" in roles and "cut_index" in roles:
        problems.append({"code": "NOT_ONE_ROW_PER_CUT", "message": "Several distinct rows share the same tool_id and cut_number.",
                         "fix": "Provide one row per cut (aggregate samples per cut first), or use the advanced column mapping for raw samples."})
    wear = roles.get("measured_wear")
    if wear and not cols[wear].get("unit"):
        problems.append({"code": "WEAR_UNIT_UNKNOWN", "message": f"'{wear}' looks like measured wear but its unit is not in the name.",
                         "fix": "Rename it to wear_mm (millimetres) or wear_um (micrometres), or choose the unit in the advanced column mapping."})
    if problems:
        raise DatasetError(422, "AUTO_MAPPING_INCOMPLETE", "This CSV could not be set up automatically.",
                           {"problems": problems, "documented_format": DOCUMENTED_FORMAT})
    mapping = {"layout": "per_cut_features",
               "columns": {c: {k: v for k, v in spec.items() if v is not None} for c, spec in cols.items()}}
    return save_mapping(dataset_id, mapping)


def _median_rows_per_cut(d: Path, m: Dict[str, Any], ti: int, ci: int) -> float:
    """Median number of distinct rows per (tool, cut); exact duplicate rows do not count as extra samples."""
    rows_per_key: Dict[Tuple[str, str], set] = {}
    with contextlib.closing(_iter_rows(d / "original.csv", m["encoding"], m["delimiter"])) as it:
        next(it)
        for _, row in it:
            digest = hashlib.blake2b("".join(row).encode("utf-8"), digest_size=8).digest()
            rows_per_key.setdefault((row[ti].strip(), row[ci].strip()), set()).add(digest)
    return float(statistics.median(len(v) for v in rows_per_key.values())) if rows_per_key else 0.0


def save_mapping(dataset_id: str, mapping: Dict[str, Any]) -> Dict[str, Any]:
    d, m = _manifest(dataset_id)
    profile = _read_json(d / "profile.json")
    types = {c["name"]: c["inferred_type"] for c in profile["columns"]}
    unknown = [c for c in mapping["columns"] if c not in types]
    if unknown:
        raise DatasetError(422, "UNKNOWN_COLUMNS", "The mapping refers to columns that are not in this dataset.", {"columns": unknown[:50]})
    issues = _mapping_issues(mapping, types)
    record = {"dataset_id": dataset_id, "layout": mapping["layout"], "columns": mapping["columns"],
              "sampling_rate_hz": mapping.get("sampling_rate_hz"), "saved_at": _now(),
              "version": int(m.get("mapping_version") or 0) + 1, "issues": issues,
              "complete": not any(i["severity"] == "error" for i in issues)}
    with _lock:
        _write_json(d / "mapping.json", record)
        m["mapping_version"] = record["version"]
        m["status"] = "MAPPED" if record["complete"] else "MAPPING_INCOMPLETE"
        _write_json(d / "manifest.json", m)
    return record


def _mapping_issues(mapping: Dict[str, Any], types: Dict[str, str]) -> List[Dict[str, Any]]:
    layout, cols = mapping["layout"], mapping["columns"]
    issues: List[Dict[str, Any]] = []

    def err(code, msg, col=None, sev="error"):
        issues.append({"code": code, "severity": sev, "message": msg, "column": col})

    by_role: Dict[str, List[str]] = {}
    for col, spec in cols.items():
        by_role.setdefault(spec["role"], []).append(col)
    for role in SINGLE_ROLES:
        if len(by_role.get(role, [])) > 1:
            err("AMBIGUOUS_ROLE", f"Role '{role}' is assigned to {len(by_role[role])} columns ({', '.join(by_role[role])}); choose exactly one.")
    target_cols = by_role.get("measured_wear", []) + by_role.get("target", [])
    if len(target_cols) > 1 and len(by_role.get("measured_wear", [])) <= 1 and len(by_role.get("target", [])) <= 1:
        err("AMBIGUOUS_ROLE", f"Target / measured wear is assigned to multiple columns ({', '.join(target_cols)}); choose exactly one.")
    for role in ("tool_id", "cut_index"):
        if not by_role.get(role):
            err("REQUIRED_ROLE_MISSING", f"Map a column to '{role}'.")
    if layout == "raw_samples":
        if not by_role.get("sensor"):
            err("REQUIRED_ROLE_MISSING", "Raw-sample layout needs at least one sensor channel column.")
        if not by_role.get("timestamp") and not mapping.get("sampling_rate_hz"):
            err("REQUIRED_ROLE_MISSING", "Raw-sample layout needs a timestamp column or a declared sampling rate.")
        for col in by_role.get("feature", []):
            err("LAYOUT_ROLE_MISMATCH", f"'{col}' is mapped as a precomputed feature, which belongs to the per-cut layout.", col)
    else:
        for col in by_role.get("sensor", []):
            err("LAYOUT_ROLE_MISMATCH", f"'{col}' is mapped as a raw sensor channel, which belongs to the raw-sample layout.", col)
        if by_role.get("timestamp"):
            err("LAYOUT_ROLE_MISMATCH", "A timestamp column is only used in the raw-sample layout; it is ignored here.", by_role["timestamp"][0], "warning")

    seen_ch: Dict[str, str] = {}
    seen_ft: Dict[str, str] = {}
    for col, spec in cols.items():
        role, t = spec["role"], types[col]
        if role in ("ignore",):
            continue
        if role in NUMERIC_ROLES and t not in ("integer", "float"):
            if role == "material" and t == "text":
                err("ROLE_TYPE_MISMATCH", f"'{col}' holds text labels; materials must be numeric codes to be compared with the NASA contract.", col, "warning")
            else:
                err("ROLE_TYPE_MISMATCH", f"'{col}' is mapped as {role} but its values are {t}, not numeric.", col)
        if role == "cut_index" and t != "integer":
            err("ROLE_TYPE_MISMATCH", f"'{col}' must contain whole numbers to be a cut index (found {t}).", col)
        if role == "timestamp":
            unit = spec.get("unit")
            if not unit:
                err("UNIT_REQUIRED", f"Choose the timestamp unit for '{col}' (s, ms or ISO datetime).", col)
            elif unit == "datetime" and t != "datetime":
                err("ROLE_TYPE_MISMATCH", f"'{col}' is declared as ISO datetimes but its values are {t}.", col)
            elif unit in ("s", "ms") and t not in ("integer", "float"):
                err("ROLE_TYPE_MISMATCH", f"'{col}' is declared in {unit} but its values are {t}.", col)
        if role in ("measured_wear", "target") and not spec.get("unit"):
            err("UNIT_REQUIRED", f"Choose the unit of measured wear in '{col}' (mm or µm). It is never guessed.", col)
        if role in ("feed_rate", "depth_of_cut") and spec.get("unit", "unknown") == "unknown":
            err("UNIT_UNKNOWN", f"The unit of '{col}' is not declared; it cannot be compared with the NASA operating levels.", col, "warning")
        if role == "cutting_time" and not spec.get("unit"):
            err("UNIT_REQUIRED", f"Choose the unit of cutting time in '{col}' (min or s).", col)
        if role == "sensor":
            ch = spec.get("channel")
            if not ch:
                err("CHANNEL_REQUIRED", f"Choose which physical channel '{col}' records.", col)
            elif ch != "other":
                if ch in seen_ch:
                    err("DUPLICATE_CHANNEL", f"Channel {ch} is mapped to both '{seen_ch[ch]}' and '{col}'.", col)
                seen_ch[ch] = col
            if spec.get("unit", "unknown") == "unknown":
                err("UNIT_UNKNOWN", f"The unit of sensor column '{col}' is unknown.", col, "warning")
        if role == "feature":
            ft = spec.get("feature")
            if not ft:
                err("FEATURE_REQUIRED", f"Choose which model feature '{col}' contains, or 'other'.", col)
            elif ft != "other":
                if ft in seen_ft:
                    err("DUPLICATE_FEATURE", f"Feature {ft} is mapped to both '{seen_ft[ft]}' and '{col}'.", col)
                seen_ft[ft] = col
    return issues


# ============================================================================ validation and compatibility
def validate(dataset_id: str) -> Dict[str, Any]:
    d, m = _manifest(dataset_id)
    if not (d / "mapping.json").is_file():
        raise DatasetError(409, "MAPPING_REQUIRED", "Save a column mapping before running validation.")
    profile = _read_json(d / "profile.json")
    mapping = _read_json(d / "mapping.json")
    names = [c["name"] for c in profile["columns"]]
    idx = {n: i for i, n in enumerate(names)}
    cols = mapping["columns"]
    role_col = {spec["role"]: c for c, spec in cols.items() if spec["role"] in SINGLE_ROLES}
    layout = mapping["layout"]

    stats = _scan(d, m, mapping, idx, role_col)
    import_section = _section([dict(w) for w in profile["warnings"]], pass_detail=f"{m['n_rows']:,} rows × {m['n_columns']} columns parsed as UTF-8 CSV.")
    mapping_section = _section([dict(i) for i in mapping["issues"]],
                               pass_detail="All required roles are mapped." if mapping["complete"] else None)
    dq_issues, dq_summary = _data_quality(stats, mapping, role_col, profile)
    data_quality = _section(dq_issues, summary=dq_summary)
    fmt_issues = _format_issues(stats, layout)
    input_format = _section(fmt_issues, pass_detail=f"Rows are consistent with the {layout.replace('_', ' ')} layout.")
    compat = _compatibility(stats, mapping, role_col)
    reasons = ["Phase 1 never runs the production model on uploaded data; predictions are not implemented for custom datasets yet."]
    reasons += [f"{c['title']}: {c['detail']}" for c in compat["checks"] if c["status"] == "FAIL"]
    if compat["verdict"] == "NOT_VERIFIABLE":
        reasons.append("Precomputed features cannot be verified to come from the NASA preprocessing pipeline, so the production model cannot be applied to them.")
    statuses = [import_section["status"], mapping_section["status"], data_quality["status"], input_format["status"]]
    overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    report = {
        "dataset_id": dataset_id, "source": "CUSTOM_UPLOAD", "validated_at": _now(), "mapping_version": mapping["version"],
        "layout": layout, "overall_status": overall, "import_validity": import_section, "mapping_completeness": mapping_section,
        "data_quality": data_quality, "input_format": input_format, "model_compatibility": compat,
        "predictions": {"available": False, "model_invoked": False, "reasons": reasons},
    }
    with _lock:
        _write_json(d / "validation.json", report)
        m["status"] = "VALIDATED"
        m["validated_mapping_version"] = mapping["version"]
        m["compatibility_verdict"] = compat["verdict"]
        _write_json(d / "manifest.json", m)
    report["stale"] = False
    return report


def _section(issues: List[Dict[str, Any]], pass_detail: Optional[str] = None, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    sev = {i["severity"] for i in issues}
    status = "FAIL" if "error" in sev else "WARN" if "warning" in sev else "PASS"
    out: Dict[str, Any] = {"status": status, "issues": issues}
    if pass_detail and status != "FAIL":
        out["detail"] = pass_detail
    if summary is not None:
        out["summary"] = summary
    return out


def _scan(d: Path, m: Dict[str, Any], mapping: Dict[str, Any], idx: Dict[str, int], role_col: Dict[str, str]) -> Dict[str, Any]:
    """One streaming pass computing per-column numeric stats and per-cut structure for the mapped roles."""
    cols = mapping["columns"]
    numeric_cols = [c for c, s in cols.items() if s["role"] in NUMERIC_ROLES or (s["role"] == "timestamp" and s.get("unit") in ("s", "ms"))]
    col_stats = {c: {"n": 0, "missing": 0, "nonfinite": 0, "non_numeric": 0, "min": None, "max": None, "values": set(), "values_overflow": False}
                 for c in numeric_cols}
    ti, ci = idx.get(role_col.get("tool_id", ""), None), idx.get(role_col.get("cut_index", ""), None)
    ts_col = role_col.get("timestamp")
    ts_unit = cols[ts_col].get("unit") if ts_col else None
    wear_col = role_col.get("target") or role_col.get("measured_wear")
    cuts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    key_issues = {"tool_missing": 0, "cut_missing": 0, "cut_not_integer": 0, "cut_not_positive": 0}
    with contextlib.closing(_iter_rows(d / "original.csv", m["encoding"], m["delimiter"])) as it:
        next(it)
        for _, row in it:
            for c in numeric_cols:
                s = col_stats[c]
                s["n"] += 1
                kind, v = classify(row[idx[c]])
                if kind == "number":
                    s["min"] = v if s["min"] is None else min(s["min"], v)
                    s["max"] = v if s["max"] is None else max(s["max"], v)
                    if not s["values_overflow"]:
                        s["values"].add(v)
                        if len(s["values"]) > 50:
                            s["values_overflow"], s["values"] = True, set()
                elif kind == "missing":
                    s["missing"] += 1
                elif kind == "nonfinite":
                    s["nonfinite"] += 1
                else:
                    s["non_numeric"] += 1
            if ti is None or ci is None:
                continue
            tool, cut = row[ti].strip(), row[ci].strip()
            if classify(tool)[0] == "missing":
                key_issues["tool_missing"] += 1
                continue
            kind, v = classify(cut)
            if kind == "missing":
                key_issues["cut_missing"] += 1
                continue
            if kind != "number" or not float(v).is_integer():
                key_issues["cut_not_integer"] += 1
                continue
            if v <= 0:
                key_issues["cut_not_positive"] += 1
            g = cuts.setdefault((tool, str(int(v))), {"n": 0, "ts_first": None, "ts_last": None, "ts_missing": 0, "non_monotonic": 0, "wear": set()})
            g["n"] += 1
            if ts_col:
                tok = row[idx[ts_col]]
                t = _parse_datetime_seconds(tok) if ts_unit == "datetime" else (lambda k: k[1] if k[0] == "number" else None)(classify(tok))
                if t is None:
                    g["ts_missing"] += 1
                else:
                    if ts_unit == "ms":
                        t = t / 1000.0
                    if g["ts_last"] is not None and t <= g["ts_last"]:
                        g["non_monotonic"] += 1
                    g["ts_first"] = t if g["ts_first"] is None else g["ts_first"]
                    g["ts_last"] = t
            if wear_col and len(g["wear"]) < 3:
                kind_w, w = classify(row[idx[wear_col]])
                g["wear"].add(w if kind_w == "number" else kind_w)
    for s in col_stats.values():
        s["values"] = None if s["values_overflow"] else sorted(s["values"])
        del s["values_overflow"]
    return {"columns": col_stats, "cuts": cuts, "keys": key_issues}


def _data_quality(stats: Dict[str, Any], mapping: Dict[str, Any], role_col: Dict[str, str], profile: Dict[str, Any]):
    issues: List[Dict[str, Any]] = []
    cols = mapping["columns"]

    def add(code, msg, sev="error", col=None):
        issues.append({"code": code, "severity": sev, "message": msg, "column": col})

    k = stats["keys"]
    if k["tool_missing"]:
        add("TOOL_ID_MISSING", f"{k['tool_missing']} row(s) have no tool id and cannot be assigned to a tool.", col=role_col.get("tool_id"))
    if k["cut_missing"]:
        add("CUT_INDEX_MISSING", f"{k['cut_missing']} row(s) have no cut index.", col=role_col.get("cut_index"))
    if k["cut_not_integer"]:
        add("CUT_INDEX_NOT_INTEGER", f"{k['cut_not_integer']} row(s) have a cut index that is not a whole number.", col=role_col.get("cut_index"))
    if k["cut_not_positive"]:
        add("CUT_INDEX_NOT_POSITIVE", f"{k['cut_not_positive']} row(s) have a cut index below 1.", "warning", role_col.get("cut_index"))
    if profile["duplicate_rows"]:
        add("DUPLICATE_ROWS", f"{profile['duplicate_rows']} exact duplicate row(s).", "warning")

    for c, s in stats["columns"].items():
        role = cols[c]["role"]
        if s["non_numeric"]:
            add("NON_NUMERIC_VALUES", f"'{c}' ({role}) has {s['non_numeric']} non-numeric value(s).", "error" if role != "material" else "warning", c)
        if s["nonfinite"]:
            add("NON_FINITE_VALUES", f"'{c}' ({role}) has {s['nonfinite']} non-finite value(s).", "error", c)
        if s["missing"] and role not in ("measured_wear", "target"):
            add("MISSING_VALUES", f"'{c}' ({role}) has {s['missing']} missing value(s).", "warning", c)

    wear = None
    wc = role_col.get("target") or role_col.get("measured_wear")
    if wc and wc in stats["columns"]:
        s, unit = stats["columns"][wc], cols[wc].get("unit")
        scale = 0.001 if unit == "um" else 1.0
        measured = s["n"] - s["missing"] - s["nonfinite"] - s["non_numeric"]
        wear = {"column": wc, "unit_declared": unit, "measured_values": measured, "missing_values": s["missing"],
                "min_mm": None if s["min"] is None else round(s["min"] * scale, 6),
                "max_mm": None if s["max"] is None else round(s["max"] * scale, 6),
                "missing_policy": "Missing wear stays missing; it is never converted to zero or interpolated."}
        if s["missing"]:
            add("MEASURED_WEAR_MISSING", f"{s['missing']} row(s) have no measured wear; they remain unmeasured (null).", "info", wc)
        if s["min"] is not None and s["min"] < 0:
            add("NEGATIVE_WEAR", f"Measured wear has negative values (min {s['min']} {unit}).", "error", wc)
        if s["max"] is not None and s["max"] * scale > 2.0:
            add("WEAR_ABOVE_RANGE", f"Measured wear reaches {s['max'] * scale:.3f} mm; check the declared unit ({unit}).", "warning", wc)
        if mapping["layout"] == "raw_samples":
            varying = sum(1 for g in stats["cuts"].values() if len(g["wear"]) > 1)
            if varying:
                add("WEAR_VARIES_WITHIN_CUT", f"Measured wear changes within {varying} cut(s); flank wear is measured once per cut.", "warning", wc)

    cuts = stats["cuts"]
    tools = {t for t, _ in cuts}
    summary: Dict[str, Any] = {"tools": len(tools), "cuts": len(cuts), "measured_wear": wear}
    if mapping["layout"] == "raw_samples" and cuts:
        ns = sorted(g["n"] for g in cuts.values())
        summary["samples_per_cut"] = {"min": ns[0], "median": int(statistics.median(ns)), "max": ns[-1], "distinct_lengths": len(set(ns))}
        if role_col.get("timestamp"):
            nm = sum(1 for g in cuts.values() if g["non_monotonic"])
            miss = sum(g["ts_missing"] for g in cuts.values())
            if nm:
                add("TIMESTAMPS_NOT_INCREASING", f"Timestamps are not strictly increasing within {nm} cut(s).", "error", role_col["timestamp"])
            if miss:
                add("TIMESTAMPS_MISSING", f"{miss} sample(s) have no usable timestamp.", "error", role_col["timestamp"])
            summary["timestamp_implied_sampling_rate_hz"] = _implied_rate(cuts)
    return issues, summary


def _implied_rate(cuts: Dict[Tuple[str, str], Dict[str, Any]]) -> Optional[float]:
    rates = [(g["n"] - 1) / (g["ts_last"] - g["ts_first"]) for g in cuts.values()
             if g["n"] > 1 and g["ts_first"] is not None and g["ts_last"] is not None and g["ts_last"] > g["ts_first"] and not g["non_monotonic"]]
    return round(statistics.median(rates), 6) if rates else None


def _format_issues(stats: Dict[str, Any], layout: str) -> List[Dict[str, Any]]:
    cuts = stats["cuts"]
    if not cuts:
        return [{"code": "NO_CUTS_IDENTIFIED", "severity": "error", "column": None,
                 "message": "No rows could be assigned to a (tool, cut) pair; map tool id and cut index."}]
    multi = sum(1 for g in cuts.values() if g["n"] > 1)
    if layout == "per_cut_features" and multi:
        return [{"code": "LAYOUT_MISMATCH", "severity": "error", "column": None,
                 "message": f"{multi} (tool, cut) pair(s) occur on several rows. One-row-per-cut data cannot repeat a cut; this looks like raw samples."}]
    if layout == "raw_samples" and multi == 0:
        return [{"code": "LAYOUT_MISMATCH", "severity": "error", "column": None,
                 "message": "Every (tool, cut) pair has exactly one row. Raw-sample data has many samples per cut; this looks like one row per cut."}]
    return []


def _check(cid: str, title: str, status: str, expected: str, observed: str, detail: str) -> Dict[str, Any]:
    return {"id": cid, "title": title, "status": status, "expected": expected, "observed": observed, "detail": detail}


def _op_values(stats: Dict[str, Any], mapping: Dict[str, Any], role_col: Dict[str, str]):
    """Locate each operating-condition column (as a role, or as an explicitly mapped model feature)."""
    cols = mapping["columns"]
    found = {}
    for role, feat in OPERATING_FEATURES.items():
        col = role_col.get(role) or next((c for c, s in cols.items() if s["role"] == "feature" and s.get("feature") == feat), None)
        found[feat] = (col, stats["columns"].get(col) if col else None, cols[col] if col else None)
    return found


def _compatibility(stats: Dict[str, Any], mapping: Dict[str, Any], role_col: Dict[str, str]) -> Dict[str, Any]:
    layout, cols = mapping["layout"], mapping["columns"]
    checks: List[Dict[str, Any]] = []
    if layout == "per_cut_features":
        mapped = {s.get("feature"): c for c, s in cols.items() if s["role"] == "feature" and s.get("feature") not in (None, "other")}
        for role, feat in OPERATING_FEATURES.items():
            if role_col.get(role):
                mapped.setdefault(feat, role_col[role])
        missing = [f for f in COMBINED_FEATURES if f not in mapped]
        checks.append(_check("FEATURE_CONTRACT", "77-feature input contract", "PASS" if not missing else "FAIL",
                             f"{len(COMBINED_FEATURES)} named features in the production order", f"{len(COMBINED_FEATURES) - len(missing)} mapped",
                             "All required features are explicitly mapped." if not missing else
                             f"{len(missing)} required feature(s) are not mapped, e.g. {', '.join(missing[:6])}."))
        bad = [c for c in mapped.values() if c in stats["columns"] and (stats["columns"][c]["nonfinite"] or stats["columns"][c]["non_numeric"] or stats["columns"][c]["missing"])]
        checks.append(_check("FEATURE_VALUES", "Feature values finite and present", "PASS" if not bad else "FAIL", "finite numbers in every mapped feature",
                             f"{len(bad)} column(s) with missing / non-finite / non-numeric values",
                             "Every mapped feature value is a finite number." if not bad else "Affected: " + ", ".join(bad[:6]) + "."))
        checks.append(_check("PREPROCESSING_PROVENANCE", "Feature extraction provenance", "NOT_VERIFIABLE",
                             f"features computed by ml/prepare_data.py over each full cut at {SAMPLING_RATE_HZ:g} Hz (bands {', '.join(b for b, _, _ in SPECTRAL_BANDS)})",
                             "precomputed values supplied by the upload",
                             "Matching column names do not show that the values were computed the same way, at the same rate, from the same sensors and units."))
    else:
        sensors = {s.get("channel"): c for c, s in cols.items() if s["role"] == "sensor" and s.get("channel") not in (None, "other")}
        missing_ch = [ch for ch in NASA_CHANNELS if ch not in sensors]
        checks.append(_check("SENSOR_CHANNELS", "Six NASA sensor channels", "PASS" if not missing_ch else "FAIL",
                             ", ".join(NASA_CHANNELS), f"{len(NASA_CHANNELS) - len(missing_ch)} of {len(NASA_CHANNELS)} mapped",
                             "All six channels are mapped (identity declared by the user)." if not missing_ch else "Not mapped: " + ", ".join(missing_ch) + "."))
        units = sorted({cols[c].get("unit", "unknown") for c in sensors.values()}) or ["—"]
        checks.append(_check("SENSOR_SEMANTICS", "Sensor identity and units", "NOT_VERIFIABLE",
                             "the same physical sensors and signal chain as the NASA rig (raw outputs as stored in mill.mat)",
                             "declared units: " + ", ".join(units),
                             "Channel identity and units are declared by the user and cannot be verified from the file. The NASA channel units are themselves undocumented."))
        declared = mapping.get("sampling_rate_hz")
        cuts = stats["cuts"]
        implied = _implied_rate(cuts) if role_col.get("timestamp") else None
        rate = declared or implied
        if rate is None:
            st, det = "FAIL", "The sampling rate is unknown (no declared rate and no usable timestamps)."
        elif declared and implied and abs(declared - implied) > 0.01 * declared:
            st, det = "FAIL", f"Declared {declared:g} Hz disagrees with {implied:g} Hz implied by the timestamps."
        elif abs(rate - SAMPLING_RATE_HZ) <= 0.01 * SAMPLING_RATE_HZ:
            st, det = "PASS", f"{rate:g} Hz matches the {SAMPLING_RATE_HZ:g} Hz the spectral band features assume."
        else:
            st, det = "FAIL", f"{rate:g} Hz differs from {SAMPLING_RATE_HZ:g} Hz; the band-energy features would describe different frequencies. Resampling is not performed."
        checks.append(_check("SAMPLING_RATE", "Sampling rate", st, f"{SAMPLING_RATE_HZ:g} Hz",
                             "unknown" if rate is None else f"{rate:g} Hz ({'declared' if declared else 'from timestamps'})", det))
        ns = sorted({g["n"] for g in cuts.values()})
        ok = ns == [NASA_SAMPLES_PER_CUT]
        checks.append(_check("SEGMENT_LENGTH", "Samples per cut", "PASS" if ok else "FAIL", f"{NASA_SAMPLES_PER_CUT:,} samples per channel per cut",
                             ("none" if not ns else f"{ns[0]:,}" if len(ns) == 1 else f"{ns[0]:,}–{ns[-1]:,} ({len(ns)} distinct lengths)"),
                             "Every cut has the NASA segment length." if ok else "The NASA pipeline treats any other segment length as an invalid run (VARIABLE_LENGTH)."))
        checks.append(_check("FEATURE_EXTRACTION", "Feature extraction", "NOT_EVALUATED", "77 features from ml/prepare_data.py",
                             "not computed", "Feature extraction for uploads is a later phase; no features or predictions are computed now."))

    op = _op_values(stats, mapping, role_col)
    for feat, (col, s, spec) in op.items():
        title = {"material_id": "Material code", "feed_rate_mm_rev": "Feed rate", "depth_of_cut_mm": "Depth of cut",
                 "cutting_time_min": "Cumulative cutting time"}[feat]
        if not col:
            checks.append(_check("OPERATING_" + feat.upper(), title, "FAIL", _op_expected(feat), "not mapped",
                                 f"The production model requires {feat} as an input."))
            continue
        if s is None or s["min"] is None:
            checks.append(_check("OPERATING_" + feat.upper(), title, "FAIL", _op_expected(feat), "no numeric values",
                                 f"'{col}' has no usable numeric values."))
            continue
        unit = (spec or {}).get("unit")
        vals = s["values"]
        if feat == "cutting_time_min":
            scale = 1 / 60.0 if unit == "s" else 1.0
            lo, hi = s["min"] * scale, s["max"] * scale
            inside = NASA_CUTTING_TIME_RANGE[0] <= lo and hi <= NASA_CUTTING_TIME_RANGE[1]
            checks.append(_check("OPERATING_CUTTING_TIME_MIN", title, "PASS" if inside else "FAIL", _op_expected(feat),
                                 f"{lo:g}–{hi:g} min", "Within the trained range." if inside else "Outside the trained range; the model would extrapolate."))
            continue
        if feat in ("feed_rate_mm_rev", "depth_of_cut_mm") and role_col.get({"feed_rate_mm_rev": "feed_rate", "depth_of_cut_mm": "depth_of_cut"}[feat]) == col                 and unit in (None, "unknown"):
            checks.append(_check("OPERATING_" + feat.upper(), title, "FAIL", _op_expected(feat), "unit unknown",
                                 "The unit is not declared, so values cannot be compared with the trained levels."))
            continue
        levels = {"material_id": NASA_MATERIAL_CODES, "feed_rate_mm_rev": NASA_FEED_LEVELS, "depth_of_cut_mm": NASA_DOC_LEVELS}[feat]
        outside = None if vals is None else [v for v in vals if not any(abs(v - L) < 1e-9 for L in levels)]
        if vals is None or outside:
            obs = "more than 50 distinct values" if vals is None else ", ".join(f"{v:g}" for v in vals[:8])
            checks.append(_check("OPERATING_" + feat.upper(), title, "FAIL", _op_expected(feat), obs,
                                 "Values outside the levels the model was trained on" + (f" (e.g. {outside[0]:g})." if outside else ".")))
        else:
            note = " Codes are taken to mean the NASA materials as declared by the user." if feat == "material_id" else ""
            checks.append(_check("OPERATING_" + feat.upper(), title, "PASS", _op_expected(feat), ", ".join(f"{v:g}" for v in vals),
                                 "Values match trained levels." + note))

    checks.append(_check("INPUT_DISTRIBUTION", "Input distribution vs. training data", "NOT_EVALUATED",
                         "feature values within the training distribution", "not evaluated",
                         "Distribution checks need extracted features; they are part of the prediction phase."))
    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        verdict, summary = "INCOMPATIBLE", "The production model cannot be applied to this dataset."
    elif layout == "per_cut_features":
        verdict, summary = "NOT_VERIFIABLE", "The required columns are present, but their preprocessing cannot be verified, so the production model is not applied."
    else:
        verdict, summary = "REQUIRES_FURTHER_CHECKS", ("The declared structure matches the NASA input format. Feature extraction and distribution "
                                                       "checks (next phase) are still required before any prediction.")
    return {"model": "production Ridge wear model (artifacts/tool_wear_regressor.joblib; not loaded in this phase)",
            "contract_features": len(COMBINED_FEATURES), "verdict": verdict, "summary": summary, "checks": checks}


def _op_expected(feat: str) -> str:
    return {"material_id": "codes 1 (cast iron) or 2 (steel)", "feed_rate_mm_rev": "0.25 or 0.5 mm/rev",
            "depth_of_cut_mm": "0.75 or 1.5 mm", "cutting_time_min": "cumulative 0–105 min"}[feat]
