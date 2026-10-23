"""CNC Sentinel — measured wear progression for custom per-cut datasets (no model involved).

Reads only the uploaded file and its saved column mapping. Every value reported here is either a
measured wear value from the file or simple arithmetic on those values (differences, a least-squares
slope); nothing is predicted, interpolated or filled in. Missing measurements stay missing.
"""

from __future__ import annotations

import contextlib
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from backend import custom_datasets as cd

TREND_TOLERANCE_MM_PER_CUT = 0.001   # |slope| below this is reported as flat
DROP_TOLERANCE_MM = 0.05             # a decrease larger than this between measured cuts is flagged
MAX_PLAUSIBLE_WEAR_MM = 2.0          # values above this usually mean a unit error
JUMP_FACTOR = 3.0                    # an increase > 3x the tool's median increase (and > 0.05 mm) is flagged
DISCLAIMER = "Measured-data analysis only: values come from the uploaded file. Nothing here is a model prediction."


def _natural_key(t: str):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", t)]


def _slope(points: List[Tuple[int, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def compute_progression(dataset_id: str) -> Dict[str, Any]:
    d, m = cd._manifest(dataset_id)
    if not (d / "mapping.json").is_file():
        raise cd.DatasetError(409, "MAPPING_REQUIRED", "Set up the columns first (quick setup or column mapping).")
    mapping = cd._read_json(d / "mapping.json")
    if mapping["layout"] != "per_cut_features":
        raise cd.DatasetError(409, "PER_CUT_LAYOUT_REQUIRED",
                              "Wear progression needs one row per cut. Raw-sample files are not summarised here.")
    cols = mapping["columns"]
    role = {spec["role"]: c for c, spec in cols.items() if spec["role"] in cd.SINGLE_ROLES}
    if "tool_id" not in role or "cut_index" not in role:
        raise cd.DatasetError(409, "MAPPING_INCOMPLETE", "Map a tool_id and a cut_number column first.")
    wear_col = role.get("measured_wear")
    unit = cols[wear_col].get("unit") if wear_col else None
    scale = 0.001 if unit == "um" else 1.0
    sensor_cols = [c for c, spec in cols.items() if spec["role"] in ("feature", "feed_rate", "depth_of_cut", "cutting_time", "material")]

    profile = cd._read_json(d / "profile.json")
    names = [c["name"] for c in profile["columns"]]
    idx = {n: i for i, n in enumerate(names)}
    ti, ci, wi = idx[role["tool_id"]], idx[role["cut_index"]], idx[wear_col] if wear_col else None

    tools: Dict[str, Dict[int, Dict[str, Any]]] = {}
    issues: List[Dict[str, Any]] = []
    skipped = 0
    sensors: Dict[str, Dict[str, Any]] = {c: {"values": [], "missing": 0, "invalid": 0, "by_tool": {}} for c in sensor_cols}
    with contextlib.closing(cd._iter_rows(d / "original.csv", m["encoding"], m["delimiter"])) as it:
        next(it)
        for line, row in it:
            tool = row[ti].strip()
            kind, cut = cd.classify(row[ci])
            if cd.classify(tool)[0] == "missing" or kind != "number" or not float(cut).is_integer() or cut <= 0:
                skipped += 1
                continue
            cut = int(cut)
            entry = tools.setdefault(tool, {}).setdefault(cut, {"wear": [], "invalid": [], "missing": 0})
            if wi is not None:
                tok = row[wi]
                k, v = cd.classify(tok)
                if k == "missing":
                    entry["missing"] += 1
                elif k == "number":
                    entry["wear"].append(round(v * scale, 6))
                else:
                    entry["invalid"].append(tok.strip()[:40])
            for c in sensor_cols:
                k, v = cd.classify(row[idx[c]])
                s = sensors[c]
                if k == "number":
                    s["values"].append(v)
                    s["by_tool"].setdefault(tool, []).append((cut, v))
                elif k == "missing":
                    s["missing"] += 1
                else:
                    s["invalid"] += 1
    if skipped:
        issues.append({"code": "ROWS_WITHOUT_TOOL_OR_CUT", "severity": "warning",
                       "message": f"{skipped} row(s) have no usable tool_id or cut_number and were left out."})

    out_tools = []
    for tool in sorted(tools, key=_natural_key):
        cuts = tools[tool]
        measured: List[Tuple[int, float]] = []
        missing_cuts, unusual = [], []
        for cut in sorted(cuts):
            e = cuts[cut]
            for bad in e["invalid"]:
                unusual.append({"cut": cut, "value": bad, "reason": "Not a finite number; ignored."})
            distinct = sorted(set(e["wear"]))
            if len(distinct) > 1:
                unusual.append({"cut": cut, "value": ", ".join(f"{v:g}" for v in distinct),
                                "reason": "Conflicting wear values for the same cut; this cut is left out."})
                continue
            if not distinct:
                if wear_col:
                    missing_cuts.append(cut)
                continue
            v = distinct[0]
            if v < 0:
                unusual.append({"cut": cut, "value": f"{v:g}", "reason": "Negative wear is physically impossible; left out."})
                continue
            if v > MAX_PLAUSIBLE_WEAR_MM:
                unusual.append({"cut": cut, "value": f"{v:g}", "reason": f"Above {MAX_PLAUSIBLE_WEAR_MM} mm; check the declared unit."})
            measured.append((cut, v))
        incs = [(b[0], b[1] - a[1]) for a, b in zip(measured, measured[1:])]
        pos = [d_ for _, d_ in incs if d_ > 0]
        med_inc = statistics.median(pos) if pos else None
        for cut, dv in incs:
            if dv < -DROP_TOLERANCE_MM:
                unusual.append({"cut": cut, "value": f"{dv:+.4g} mm", "reason": f"Wear dropped by more than {DROP_TOLERANCE_MM} mm since the previous measured cut."})
            elif med_inc and dv > max(DROP_TOLERANCE_MM, JUMP_FACTOR * med_inc) and len(pos) >= 3:
                unusual.append({"cut": cut, "value": f"{dv:+.4g} mm", "reason": "Unusually large increase compared with this tool's other increases."})
        slope = _slope(measured)
        if len(measured) < 2 or slope is None:
            trend = "INSUFFICIENT_DATA"
        elif slope > TREND_TOLERANCE_MM_PER_CUT:
            trend = "INCREASING"
        elif slope < -TREND_TOLERANCE_MM_PER_CUT:
            trend = "DECREASING"
        else:
            trend = "FLAT"
        first, last = (measured[0], measured[-1]) if measured else (None, None)
        out_tools.append({
            "tool_id": tool, "n_cuts": len(cuts), "cuts": sorted(cuts), "measured": [{"cut": c, "wear_mm": v} for c, v in measured],
            "measured_count": len(measured), "missing_cuts": missing_cuts,
            "first_cut": first[0] if first else None, "first_wear_mm": first[1] if first else None,
            "latest_cut": last[0] if last else None, "latest_wear_mm": last[1] if last else None,
            "change_mm": round(last[1] - first[1], 6) if measured and len(measured) >= 2 else None,
            "rate_mm_per_cut": None if slope is None else round(slope, 6), "trend": trend,
            "unusual": sorted(unusual, key=lambda u: u["cut"]),
        })

    has_wear = wear_col is not None
    comparison = None
    if has_wear:
        rated = [t for t in out_tools if t["rate_mm_per_cut"] is not None]
        if len(rated) >= 2:
            by_rate = sorted(rated, key=lambda t: t["rate_mm_per_cut"], reverse=True)
            with_latest = [t for t in out_tools if t["latest_wear_mm"] is not None]
            comparison = {
                "ranking_by_rate": [{"tool_id": t["tool_id"], "rate_mm_per_cut": t["rate_mm_per_cut"], "latest_wear_mm": t["latest_wear_mm"]} for t in by_rate],
                "fastest_tool": by_rate[0]["tool_id"], "slowest_tool": by_rate[-1]["tool_id"],
                "median_rate_mm_per_cut": round(statistics.median(t["rate_mm_per_cut"] for t in rated), 6),
                "highest_latest_tool": max(with_latest, key=lambda t: t["latest_wear_mm"])["tool_id"] if with_latest else None,
            }
        total_missing = sum(len(t["missing_cuts"]) for t in out_tools)
        if total_missing:
            issues.append({"code": "MISSING_MEASUREMENTS", "severity": "info",
                           "message": f"{total_missing} cut(s) have no wear measurement; they are shown as gaps, never filled in."})
        n_unusual = sum(len(t["unusual"]) for t in out_tools)
        if n_unusual:
            issues.append({"code": "UNUSUAL_VALUES", "severity": "warning", "message": f"{n_unusual} unusual wear value(s) need review."})
        rising = [t["tool_id"] for t in out_tools if t["trend"] == "INCREASING"]
        if rising:
            issues.append({"code": "INCREASING_WEAR", "severity": "info",
                           "message": f"Measured wear is increasing for {len(rising)} tool(s): {', '.join(rising[:10])}."})

    sensor_summary = []
    for c in sensor_cols:
        s = sensors[c]
        vals = s["values"]
        changes = []
        for tool, pts in s["by_tool"].items():
            pts = sorted(pts)
            if len(pts) >= 2:
                changes.append(pts[-1][1] - pts[0][1])
        sensor_summary.append({
            "column": c, "role": cols[c]["role"], "count": len(vals), "missing": s["missing"], "invalid": s["invalid"],
            "min": min(vals) if vals else None, "max": max(vals) if vals else None,
            "mean": round(statistics.fmean(vals), 6) if vals else None,
            "tools_rising": sum(1 for x in changes if x > 0), "tools_falling": sum(1 for x in changes if x < 0), "tools_compared": len(changes),
        })

    return {
        "dataset_id": dataset_id, "mapping_version": mapping["version"], "analysis_type": "MEASURED_DATA_ANALYSIS",
        "disclaimer": DISCLAIMER, "has_wear": has_wear, "wear_column": wear_col, "wear_unit_declared": unit,
        "no_wear_explanation": None if has_wear else "No measured wear column is mapped, so wear progression cannot be calculated. Sensor columns can still be inspected; no wear values are estimated.",
        "tool_count": len(out_tools), "tools": out_tools, "comparison": comparison, "issues": issues,
        "sensor_summary": sensor_summary,
        "rules": {"trend_tolerance_mm_per_cut": TREND_TOLERANCE_MM_PER_CUT, "drop_tolerance_mm": DROP_TOLERANCE_MM,
                  "max_plausible_wear_mm": MAX_PLAUSIBLE_WEAR_MM, "jump_factor": JUMP_FACTOR,
                  "rate_method": "least-squares slope of measured wear against cut number"},
    }
