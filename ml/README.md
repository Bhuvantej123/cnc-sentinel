# ML Pipeline — CNC Sentinel

Machine learning pipeline for CNC cutting-tool wear estimation, uncertainty quantification, and Remaining Useful Life (RUL) forecasting.

---

## Assigned Contributor: P2 (ML Engineer)

### Primary Dataset: NASA Milling Dataset
- Raw files placed in `../data/raw/`.
- 16 tool cases with intermittent optical microscope measurements of flank wear ($V_B$).
- Sensor channels to process: Acoustic emission (`AE_table`, `AE_spindle`), vibration (`vib_table`, `vib_spindle`), and motor currents (`smcAC`, `smcDC`).

### Core Stack:
- `numpy==1.26.2`, `pandas==2.1.4`: Data manipulation and tabular aggregations.
- `scipy==1.11.4`: Digital filtering, time-domain moments, and power spectral density (PSD) computation.
- `scikit-learn==1.3.2`: Feature scaling, regressor models (RandomForestRegressor, GradientBoostingRegressor, Ridge), quantile regression for uncertainty bounds, and Leave-One-Tool-Out (LOTO) evaluation.
- `joblib==1.3.2`: Serializing trained model pipelines into `../artifacts/`.

---

## Suggested Pipeline Workflow:

1. **`prepare_data.py`**:
   - Parse cut records from `../data/raw/` (e.g. `mill.mat` or extracted MAT/CSV cuts).
   - Compute statistical features per cut (Mean, RMS, Peak, Kurtosis, Skewness, Crest Factor, Band Energy).
   - Export per-cut feature table directly to `../artifacts/cut_features.csv` following the contract in `../artifacts/README.md`.
   - Preserve intermittent measurements: cuts without physical wear inspections must have empty/NaN `measured_wear_vb`.
2. **`train.py`**:
   - Train tool wear regression model on training cuts.
   - Fit uncertainty estimation (e.g., residual standard deviation or quantile regressors).
   - Save `tool_wear_regressor.joblib`, `feature_scaler.joblib`, and `feature_names.json` into `../artifacts/`.
3. **`evaluate.py`**:
   - Run Leave-One-Tool-Out (LOTO) cross-validation across the 16 cases to assess generalization.
   - Save summary metrics ($R^2$, RMSE, MAE, RUL error) into `../artifacts/metrics_report.json`.
4. **`inference.py`**:
   - Single point of entry for live predictions and RUL calculation (`predict(tool_id, cut_index, wear_threshold, risk_tolerance)`).

> [!CAUTION]
> Do not generate fake predictions or commit large binary `.joblib` or raw dataset files to Git.
