"""Simplified custom workflow: quick setup (auto mapping) and measured wear progression without any model.

Fixtures are small synthetic CSVs written inline. The NASA model is patched to fail loudly, so any
use of it during upload, quick setup, progression, readiness or optional training fails the test.
"""

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import custom_datasets as cd
from backend.main import app

client = TestClient(app)

DOC = ("tool_id,cut_number,wear_mm,spindle_speed,feed_rate,cutting_force,vibration,temperature\n"
       "T1,1,0.02,1200,0.1,410,0.31,35.0\n"
       "T1,2,0.05,1200,0.1,425,0.33,36.1\n"
       "T1,3,0.08,1200,0.1,440,0.36,37.0\n"
       "T1,4,0.11,1200,0.1,455,0.40,38.2\n"
       "T2,1,0.01,1500,0.2,390,0.28,34.0\n"
       "T2,2,,1500,0.2,392,0.29,34.4\n"
       "T2,3,0.03,1500,0.2,401,0.30,35.0\n"
       "T2,4,0.04,1500,0.2,405,0.30,35.3\n")


@pytest.fixture
def store(tmp_path: Path, monkeypatch) -> Path:
    import ml.inference as inf

    def nasa_model_used(*a, **k):
        raise AssertionError("custom data must never reach the NASA production model")

    root = tmp_path / "custom"
    monkeypatch.setattr(cd, "STORAGE_ROOT", root)
    for name in ("predict", "validate_model_artifact", "is_model_verified"):
        monkeypatch.setattr(inf, name, nasa_model_used)
    monkeypatch.setattr(inf, "MODEL_PATH", tmp_path / "nasa_model_must_not_be_read.joblib")
    return root


def _upload(text):
    r = client.post("/api/datasets", params={"filename": "shop.csv"}, content=text.encode())
    assert r.status_code == 201, r.text
    return r.json()["manifest"]["dataset_id"]


def _auto(ds):
    return client.post(f"/api/datasets/{ds}/mapping/auto")


def _progression(ds):
    r = client.get(f"/api/datasets/{ds}/progression")
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------------ quick setup
def test_documented_format_is_recognised_without_manual_mapping(store):
    ds = _upload(DOC)
    r = _auto(ds)
    assert r.status_code == 200, r.text
    rec = r.json()
    cols = rec["columns"]
    assert rec["complete"] and rec["layout"] == "per_cut_features"
    assert cols["tool_id"]["role"] == "tool_id" and cols["cut_number"]["role"] == "cut_index"
    assert cols["wear_mm"] == {"role": "measured_wear", "unit": "mm", "channel": None, "feature": None}
    assert cols["feed_rate"]["role"] == "feed_rate"
    for f in ("spindle_speed", "cutting_force", "vibration", "temperature"):
        assert cols[f]["role"] == "feature" and cols[f]["feature"] == "other"
    assert sum(spec["role"] == "measured_wear" for spec in cols.values()) == 1


def test_wear_like_columns_are_never_auto_used_as_features(store):
    ds = _upload("tool_id,cut_number,wear_mm,predicted_wear,vb_estimate,force\nA,1,0.1,0.1,0.1,5\nA,2,0.2,0.2,0.2,6\n")
    rec = _auto(ds).json()
    assert "predicted_wear" not in rec["columns"] and "vb_estimate" not in rec["columns"]
    assert rec["columns"]["force"]["role"] == "feature"
    notes = client.get(f"/api/datasets/{ds}/mapping/suggestions").json()["notes"]
    assert any("predicted_wear" in n and "not used as an input feature" in n for n in notes)


@pytest.mark.parametrize("csv_text,code,fix_word", [
    ("tool,tool_id,cut_number,wear_mm\nA,A,1,0.1\nA,A,2,0.2\n", "AMBIGUOUS_COLUMNS", "Keep exactly one"),
    ("tool_id,wear_mm,force\nA,0.1,5\nA,0.2,6\n", "MISSING_CUT_NUMBER", "cut_number"),
    ("cut_number,wear_mm,force\n1,0.1,5\n2,0.2,6\n", "MISSING_TOOL_ID", "tool_id"),
    ("tool_id,cut_number,wear,force\nA,1,0.1,5\nA,2,0.2,6\n", "WEAR_UNIT_UNKNOWN", "wear_mm"),
    ("tool_id,cut_number,force\nA,1,5\nA,1,6\nA,2,7\nA,2,8\n", "NOT_ONE_ROW_PER_CUT", "one row per cut"),
])
def test_quick_setup_explains_how_to_fix_the_csv(store, csv_text, code, fix_word):
    ds = _upload(csv_text)
    r = _auto(ds)
    assert r.status_code == 422
    det = r.json()["detail"]
    assert det["error_code"] == "AUTO_MAPPING_INCOMPLETE" and "tool_id" in det["details"]["documented_format"]
    probs = {p["code"]: p for p in det["details"]["problems"]}
    assert code in probs and fix_word in probs[code]["fix"]
    assert not (store / ds / "mapping.json").exists()  # nothing saved when setup is incomplete


# ------------------------------------------------------------------ progression
def test_progression_per_tool_values_and_comparison(store):
    ds = _upload(DOC)
    _auto(ds)
    p = _progression(ds)
    assert p["analysis_type"] == "MEASURED_DATA_ANALYSIS" and "nothing here is a model prediction" in p["disclaimer"].lower()
    assert p["has_wear"] and p["tool_count"] == 2
    t1, t2 = p["tools"]
    assert t1["tool_id"] == "T1" and [m["wear_mm"] for m in t1["measured"]] == [0.02, 0.05, 0.08, 0.11]
    assert t1["latest_cut"] == 4 and t1["latest_wear_mm"] == 0.11 and t1["change_mm"] == pytest.approx(0.09)
    assert t1["rate_mm_per_cut"] == pytest.approx(0.03) and t1["trend"] == "INCREASING"
    assert t2["missing_cuts"] == [2] and [m["cut"] for m in t2["measured"]] == [1, 3, 4]  # gap kept, never filled
    assert t2["rate_mm_per_cut"] == pytest.approx(0.01, abs=1e-6)
    c = p["comparison"]
    assert c["fastest_tool"] == "T1" and c["slowest_tool"] == "T2" and c["highest_latest_tool"] == "T1"
    codes = {i["code"] for i in p["issues"]}
    assert {"MISSING_MEASUREMENTS", "INCREASING_WEAR"} <= codes
    assert {s["column"] for s in p["sensor_summary"]} == {"spindle_speed", "feed_rate", "cutting_force", "vibration", "temperature"}
    assert "wear_mm" not in {s["column"] for s in p["sensor_summary"]}


def test_invalid_and_unusual_wear_values_are_flagged_not_used(store):
    text = ("tool_id,cut_number,wear_mm,force\n"
            "A,1,0.10,5\nA,2,0.20,5\nA,3,0.05,5\nA,4,-0.01,5\nA,5,abc,5\nA,6,inf,5\nA,7,2.6,5\n"
            "B,1,0.10,5\nB,1,0.12,5\nB,2,0.15,5\n")
    ds = _upload(text)
    assert _auto(ds).status_code == 200
    a, b = _progression(ds)["tools"]
    reasons = {u["cut"]: u["reason"] for u in a["unusual"]}
    assert "dropped" in reasons[3] and "Negative" in reasons[4] and "finite number" in reasons[5] and "finite number" in reasons[6]
    assert "check the declared unit" in reasons[7]
    assert [m["cut"] for m in a["measured"]] == [1, 2, 3, 7]  # negative / non-numeric excluded, nothing zero-filled
    assert a["missing_cuts"] == [5, 6]
    assert any("Conflicting" in u["reason"] and u["cut"] == 1 for u in b["unusual"])
    assert [m["cut"] for m in b["measured"]] == [2] and b["trend"] == "INSUFFICIENT_DATA"


def test_wear_in_micrometres_converted_only_when_named(store):
    ds = _upload("tool_id,cut_number,wear_um,force\nA,1,50,5\nA,2,120,6\n")
    assert _auto(ds).json()["columns"]["wear_um"]["unit"] == "um"
    t = _progression(ds)["tools"][0]
    assert [m["wear_mm"] for m in t["measured"]] == [0.05, 0.12]


def test_without_wear_only_sensor_inspection_and_no_invented_values(store):
    ds = _upload("tool_id,cut_number,spindle_speed,vibration\nA,1,1000,0.2\nA,2,1000,0.3\nB,1,900,0.1\nB,2,900,0.1\n")
    assert _auto(ds).status_code == 200
    p = _progression(ds)
    assert p["has_wear"] is False and "cannot be calculated" in p["no_wear_explanation"]
    assert all(t["measured"] == [] and t["latest_wear_mm"] is None and t["missing_cuts"] == [] for t in p["tools"])
    assert p["comparison"] is None
    vib = next(s for s in p["sensor_summary"] if s["column"] == "vibration")
    assert vib["count"] == 4 and vib["tools_rising"] == 1 and vib["tools_compared"] == 2


def test_progression_requires_a_per_cut_mapping(store):
    ds = _upload(DOC)
    r = client.get(f"/api/datasets/{ds}/progression")
    assert r.status_code == 409 and r.json()["detail"]["error_code"] == "MAPPING_REQUIRED"


# ------------------------------------------------------------------ optional training stays separate and leak-free
def test_optional_training_never_uses_wear_as_input_and_never_touches_nasa_model(store):
    rows = ["tool_id,cut_number,wear_mm,spindle_speed,cutting_force,vibration"]
    for k, t in enumerate(("A", "B", "C", "D")):
        for c in range(1, 5):
            w = 0.02 + (0.03 + 0.004 * k) * (c - 1)
            rows.append(f"{t},{c},{w:.4f},{1000 + 50 * k},{400 + 300 * w + c},{0.2 + w + 0.01 * k}")
    ds = _upload("\n".join(rows) + "\n")
    assert _auto(ds).status_code == 200
    assert _progression(ds)["has_wear"]  # insights work without any model
    rdy = client.get(f"/api/datasets/{ds}/training-readiness").json()
    assert "wear_mm" not in rdy["usable_features"]
    assert rdy["status"] in ("READY_FOR_EXPERIMENT", "NEEDS_REVIEW"), rdy
    model = client.post(f"/api/datasets/{ds}/train", json={"alpha": 10.0})
    assert model.status_code == 200, model.text
    assert "wear_mm" not in model.json()["feature_names"] and model.json()["target_column"] == "wear_mm"
    assert set(model.json()["feature_names"]) == {"spindle_speed", "cutting_force", "vibration"}
