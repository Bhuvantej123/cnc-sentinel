# CNC Sentinel — 24-Hour Project Execution Plan

## 1. Project Purpose & Problem Statement
In CNC machining, cutting tools (end mills, inserts) undergo progressive flank wear ($V_B$) due to friction, thermal stress, and mechanical abrasion. Unmonitored tool wear leads to:
1. Catastrophic tool breakage, damaging expensive raw workpieces and spindle bearings.
2. Out-of-tolerance surface finishes and scrapped parts.
3. Premature tool replacement, creating unnecessary tooling costs.

**CNC Sentinel** is an AI-powered system that predicts flank wear progression and estimates **Remaining Useful Life (RUL)** in cutting passes, providing operators and shop managers with actionable, risk-calibrated recommendations (`CONTINUE`, `INSPECT_SOON`, `REPLACE_TOOL`).

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph UI ["Frontend (React + Vite + Tailwind) - Controlled by P1"]
        A[Tool Selector & Cut Slider] --> B[Wear Curve & RUL Display]
        B --> C[Operator Recommendation Badge]
        B --> D[Uncertainty Band Visualizer]
    end

    subgraph API ["Backend (FastAPI) - P3 & P1"]
        E[GET /api/health]
        F[GET /api/meta]
        G[GET /api/tools]
        H[GET /api/tools/{tool_id}/history]
        I[POST /api/predict]
        J[GET /api/generalization]
        I --> K[ml.inference.predict]
    end

    subgraph ML_MOD ["Machine Learning Pipeline - P2"]
        L[NASA Milling Dataset] --> M[Feature Extraction: Time & Spectral]
        M --> N[artifacts/cut_features.csv]
        N --> O[Wear Regressor & Quantile Estimator]
        O --> P[artifacts/tool_wear_regressor.joblib]
        O --> Q[artifacts/metrics_report.json]
    end

    UI <-->|Vite Proxy /api| API
    P -.->|Loaded & validated| K
    N -.->|Tool discovery| G
    Q -.->|Serves metrics| J
```

---

## 3. Team Responsibilities & Boundaries

The workflow is structured so beginner teammates have bounded, well-defined tasks, while P1 controls the core motion system, architecture, and integration:

| Role | Team Member | Primary Scope | Deliverables |
| :--- | :--- | :--- | :--- |
| **P1** | **Lead & Integrator** | System architecture, backend/model handoff, integration testing, and the **advanced animated frontend experience** (main motion system). | Core repo foundation, motion system, end-to-end integration, demo validation. |
| **P2** | **ML Engineer (Beginner-Friendly)** | Preprocessing the NASA Milling dataset, extracting per-cut tabular features, and fitting a scikit-learn baseline model. | `ml/prepare_data.py`, `artifacts/cut_features.csv`, `ml/train.py`, `artifacts/metrics_report.json`. |
| **P3** | **Backend Developer (Beginner-Friendly)** | Implementing FastAPI endpoints matching the contract, discovering tools from `cut_features.csv`, and delegating predictions to `ml/inference.py`. | `backend/main.py` routes, Pydantic schemas, history endpoint. |
| **P4** | **Demo & Presentation Lead** | Owns slide deck (PPT), demo pitch narrative, user story, video recording, and supporting UI assets. | Pitch deck, 2-minute demo flow script, judging presentation. |

---

## 4. 24-Hour Hackathon Timeline Breakdown

| Time Window | Phase | Key Objectives | Milestones |
| :--- | :--- | :--- | :--- |
| **Hours 0 – 3** | **Foundation & Architecture** | Repository structure, git policies, API contract specification, minimal backend skeleton, dataset criteria. | Foundation locked, team aligned on `shared/api_contract.md`. |
| **Hours 3 – 8** | **Core Development: Baseline** | • **ML (P2)**: Parse NASA Milling dataset; extract RMS, peak, kurtosis; write `artifacts/cut_features.csv`.<br>• **Backend (P3)**: Implement `/api/tools`, `/api/tools/{tool_id}/history`, and `/api/meta` with dynamic discovery.<br>• **Frontend (P1)**: Establish Vite proxy, layout, motion system, and theme tokens.<br>• **Demo (P4)**: Draft problem statement, architecture slide, and business value metrics. | Backend serves dynamic tool list from `cut_features.csv`. Frontend shell running. |
| **Hours 8 – 14** | **Model & Logic Integration** | • **ML (P2)**: Train wear regressor on training cuts, fit uncertainty bounds, serialize `.joblib`.<br>• **Backend (P3/P1)**: Connect `POST /api/predict` via `ml/inference.py` with runtime validation.<br>• **Frontend (P1)**: Wear curve with confidence bands, RUL counter, and risk tolerance selector.<br>• **Demo (P4)**: Build operator walkthrough flow. | Live prediction responses verified against real NASA cuts. |
| **Hours 14 – 19** | **Refinement & Polish** | • **ML (P2)**: Compute Leave-One-Tool-Out (LOTO) metrics for `/api/generalization`.<br>• **Backend (P3)**: Error handling, edge-case validation, CORS & proxy verification.<br>• **Frontend (P1)**: Polished micro-animations, glassmorphic metric cards, state transitions.<br>• **Demo (P4)**: Rehearse presentation timing, create demo screenshots and visual aids. | High visual appeal, zero mock values in live mode. |
| **Hours 19 – 22** | **End-to-End Hardening** | Full team smoke tests. P1 validates contract adherence, missing file fallbacks, and network resilience. | Frozen codebase, zero unhandled errors. |
| **Hours 22 – 24** | **Pitch Practice & Submission** | P4 leads demo rehearsal. Final slide deck export and repository cleanup. | Project submitted. |

---

## 5. Risk Assessment & Mitigations

1. **Risk: ML training takes too long or overfits.**
   - *Mitigation*: Stick to tabular feature extraction (scipy / numpy) + scikit-learn regressors. Avoid deep neural networks requiring GPU acceleration.
2. **Risk: Teammates block each other.**
   - *Mitigation*: `artifacts/cut_features.csv` is the clear boundary between ML and Backend. Frontend uses Vite proxy with relative API endpoints, and can use local dev fixtures (`src/fixtures/`) until the backend is online.
3. **Risk: Secret leaks or huge repo size.**
   - *Mitigation*: Strict `.gitignore` already configured for data files and weights.
