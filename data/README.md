# CNC Sentinel — Dataset Directory

This directory is designated for raw and preprocessed datasets used in cutting-tool flank wear prediction and Remaining Useful Life (RUL) estimation.

> [!NOTE]
> All raw data files (`.csv`, `.mat`, `.zip`, `.parquet`) in this directory are excluded from Git version control via `.gitignore` to keep the repository lightweight and prevent large binary commits.

---

## Primary Candidate Dataset: NASA Milling Dataset

The project adopts the **NASA Milling Dataset** as its primary candidate source for model training and validation:
- **Test Setup**: Acoustic emission (AE), table and spindle vibration, and spindle motor current signals recorded during face milling cuts on a Matsuura machining center.
- **Run Matrix**: 16 tool cases with variations in workpiece material (cast iron and steel), feed rate (0.25 mm/rev and 0.5 mm/rev), and depth of cut (0.75 mm and 1.5 mm).
- **Flank Wear ($V_B$)**: Flank wear was periodically measured using an optical microscope.

### Mandatory Verification Before Modeling
The ML team must strictly inspect the real dataset files upon receipt. Do **not** assume or hard-code values:
1. **Sensor Channels**: Verify sampling frequency, signal units, and channel mapping (e.g. `smcAC`, `smcDC`, `vib_table`, `vib_spindle`, `AE_table`, `AE_spindle`).
2. **Wear Measurement Units**: Verify whether flank wear $V_B$ is recorded in millimeters (mm) or micrometers ($\mu$m). Typical NASA records store $V_B$ in mm (ranging from ~0.05 mm to 0.45 mm).
3. **Intermittent Wear Labels**: Optical microscope measurements were not taken at every cut pass. Many cut passes have missing flank wear values. These must be represented as `NaN` / `null`—never interpolated without explicit documentation or fabricated.
4. **Operating Conditions**: Confirm speed (RPM), feed rate, and material mapping directly from the dataset run records.
5. **Wear Threshold**: Establish a domain-justified tool retirement threshold (e.g., $V_B = 0.20\text{ mm}$ or $0.30\text{ mm}$) based on the verified dataset bounds. Do not invent arbitrary thresholds.

---

## Directory Organization

```text
data/
├── README.md               # This documentation
├── raw/                    # Untouched dataset files (e.g., mill.mat or extracted cut files)
│   └── (local only, git-ignored)
└── processed/              # Extracted cut-level features (e.g., intermediate tabular data)
    └── (local only, git-ignored)
```

---

## Data Hygiene Guidelines for Teammates
1. **Never commit raw sensor files or large archives** to Git.
2. Verify all extracted cut summaries retain explicit units.
3. Do not impute or alter ground-truth tool wear values without logging the transformation in `ml/`.
4. Extracted cut features destined for the backend must follow the artifact contract in `../artifacts/README.md`.
