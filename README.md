
# Cognitive Load Pipeline (EEG, PPG, GSR)

> This repo uses a YAML config. Copy `configs/config.example.yaml` to `configs/config.yaml` and set your local directories.

## Quickstart

```bash
# 1) Create a virtual env (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure paths
cp configs/config.example.yaml configs/config.yaml
# ...edit configs/config.yaml to point to your local data folders

# 4) Run pipelines
python scripts/run_gsr.py --config configs/config.yaml
python scripts/run_ppg.py --config configs/config.yaml
python scripts/run_eeg.py --config configs/config.yaml

# 5) Multimodal fusion (PPG+GSR)
python scripts/run_fusion.py --config configs/config.yaml

# 6) Benchmark from any saved CSVs
python scripts/benchmark.py   --features outputs/EEGfeatures.csv   --labels outputs/EEGlabels.csv
```

### Feature Importance (Random Forest MDI)
```bash
python scripts/feature_importance.py   --features outputs/PPGfeatures.csv   --labels outputs/PPGlabels.csv   --out outputs/ppg_feature_importance.png   --title "PPG Feature Importances"
```

> You can also run it for GSR/EEG by changing the inputs.

---

## Expected Data Layouts

### PPG/GSR (CLAS)
- `block_details_dir` contains per-user CSVs with columns like **`Block`** and **`EDA&PPG File`**.
- `participants_dir` contains per-user directories: `<user_id>/by_block/<csv>`, where each CSV has columns `ppg`, `gsr`.

Special-case file renames 
- Block 2 → append `mathtest` before file extension
- Block 8 → append `IQtest` before file extension

### EEG
- `eeg_root` contains per-user directories, each with files named like `...data_level{N}.csv` (N = level).
- `labels_root` has `<user_id>.csv` with columns `lvl_{N}` containing integer ratings 1–9 which we map to `low` (1–3), `medium` (4–6), `high` (7–9).

---

## Outputs

All scripts save CSVs and artifacts to `outputs/` by default:
- `GSRfeatures.csv`, `GSRlabels.csv`, `GSRfeatures_with_labels.csv`
- `PPGfeatures.csv`, `PPGlabels.csv`, `PPGfeatures_with_labels.csv`
- `EEGfeatures.csv`, `EEGlabels.csv`, `EEGfeatures_with_labels.csv`

---
---

## Repo Structure

```
clas-cognitive-load/
├── configs/
│   ├── config.example.yaml
├── notebooks/
├── outputs/                  # created at runtime
├── scripts/
│   ├── run_gsr.py
│   ├── run_ppg.py
│   ├── run_eeg.py
│   ├── run_fusion.py
│   ├── benchmark.py
│   ├── feature_importance.py
├── src/
│   └── clascl/
│       ├── __init__.py
│       ├── utils.py
│       ├── models.py
│       ├── gsr.py
│       ├── ppg.py
│       ├── eeg.py
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Citation / Credit
- feature extraction uses `neurokit2` and `EEGEXTRACT` utilities.
