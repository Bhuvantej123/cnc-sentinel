# CNC Sentinel

> **AI-Powered CNC Cutting-Tool Wear Prediction & Remaining Useful Life (RUL) Estimation**
> *24-Hour Hackathon Project Foundation*

---

## 1. Project Purpose & Vision

In precision CNC milling, tool wear is an inevitable physical degradation process. If undetected, excessive flank wear ($V_B$) causes tool chipping, workpiece surface defects, and emergency spindle shutdowns. Conversely, replacing cutting tools strictly on fixed conservative schedules wastes up to 30% of tool life.

**CNC Sentinel** bridges this gap:
- Analyzes multi-sensor machining signatures (vibration, acoustic emissions, spindle current).
- Predicts continuous cutting-tool flank wear ($V_B$).
- Projects Remaining Useful Life (RUL) in remaining allowable passes/cuts.
- Provides risk-aware operator advisories (`CONTINUE`, `INSPECT_SOON`, `REPLACE_TOOL`).

---

## 2. Primary Dataset Decision: NASA Milling Dataset

The project adopts the **NASA Milling Dataset** as its primary candidate source for model development:
- **Structure**: 16 tool cases with varying workpiece materials, feeds, and depths of cut.
- **Sensor Signals**: Acoustic emission, table/spindle accelerometer vibration, and motor currents.
- **Strict Verification Rule**: Sensor channels, units, labels, operating conditions, and wear threshold ($V_B$ limit in mm) must be verified directly against the actual dataset files. Measurements in this dataset are taken intermittently; missing wear labels are represented as `null`/`NaN` and never fabricated.

---

## 3. System Architecture & Tech Stack

```text
cnc-sentinel/
├── frontend/             # React 18 + TypeScript + Vite + Tailwind CSS (controlled by P1)
├── backend/              # Python 3.10+ + FastAPI + Uvicorn + Pydantic (P3 & P1)
├── ml/                   # pandas, NumPy, SciPy, scikit-learn (P2)
├── data/                 # Raw & processed NASA Milling runs (git-ignored)
├── artifacts/            # Serialized cut_features.csv, models, scalers (git-ignored)
├── shared/               # Cross-team interfaces & api_contract.md
├── docs/                 # Hackathon execution plan & architecture docs
├── .gitignore            # Security, data, node_modules & artifact ignores
└── README.md             # Project master overview
```

### Technology Matrix
- **Frontend**: React 18, TypeScript, Vite (with `/api` dev proxy), Tailwind CSS v3, Lucide icons, Recharts.
- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2.
- **Machine Learning**: pandas, NumPy, SciPy (spectral & time features), scikit-learn (wear regression and quantile bounds).
- **Interface Contract**: [shared/api_contract.md](shared/api_contract.md).

---

## 4. Team Responsibilities & Workflow

| Role | Focus Area | Deliverables |
| :--- | :--- | :--- |
| **P1 (Lead & Integrator)** | System architecture, backend/model handoff, integration, and the **advanced animated frontend experience** | Foundation, motion system, end-to-end integration, demo validation |
| **P2 (ML Engineer)** | Data ingestion, signal feature extraction, wear regression, RUL estimation | `ml/prepare_data.py`, `artifacts/cut_features.csv`, `ml/train.py` |
| **P3 (Backend Developer)** | FastAPI endpoints, Pydantic models, dynamic tool discovery from `cut_features.csv` | `backend/main.py`, route modules, history endpoint |
| **P4 (Demo & Pitch Lead)** | Presentation deck (PPT), demo pitch narrative, user story, and pitch script | Pitch slide deck, 2-minute demo flow script |

---

## 5. Quick Start & Setup Guide

### 5.1 Backend Setup (P3 & P1)
```bash
# 1. Navigate to backend
cd backend

# 2. Create and activate a Python virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 3. Install backend dependencies
pip install -r requirements.txt

# 4. Start the development server
uvicorn main:app --reload --port 8000
```
- Interactive Swagger docs: `http://localhost:8000/docs`
- Health check endpoint: `http://localhost:8000/api/health`

### 5.2 ML Pipeline Setup (P2)
```bash
# 1. Navigate to ml
cd ml

# 2. Install ML environment dependencies
pip install -r requirements.txt

# 3. Follow data/README.md to place NASA Milling files in data/
```

### 5.3 Frontend Setup (P1 & P4)
```bash
# 1. Navigate to frontend
cd frontend

# 2. Initialize Vite + React + TS with Tailwind CSS v3
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss@^3.4.1 postcss autoprefixer lucide-react recharts
npx tailwindcss init -p

# 3. Configure Vite proxy for /api and run dev server
npm run dev
```

---

## 6. API Endpoints Overview

Detailed JSON schemas and response models are specified in [`shared/api_contract.md`](shared/api_contract.md):

- `GET  /api/health`: Service status and model load verification (requires validated estimator interface).
- `GET  /api/meta`: Tool limits, wear unit, supported sensors, and system metadata.
- `GET  /api/tools`: Dynamically discovered cutting tools from `artifacts/cut_features.csv`.
- `GET  /api/tools/{tool_id}/history`: Cut-by-cut measured wear vs. prediction history.
- `POST /api/predict`: Tool wear prediction, uncertainty interval, and remaining cuts via `ml/inference.py`.
- `GET  /api/generalization`: Cross-tool Leave-One-Tool-Out (LOTO) validation scores.

> [!IMPORTANT]
> **Data & Model Integrity Policy**: Model outputs and metrics remain flagged as `UNAVAILABLE` or `null` until the real dataset is processed and verified models are trained. Fabricated or mock model predictions are strictly prohibited from the backend.

---

## 7. Project Roadmap

Refer to [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the complete hour-by-hour 24-hour sprint plan.
