# NASA Turbofan RUL — ML Project (CMAPSS)

End-to-end machine learning project to predict **Remaining Useful Life (RUL)** on the NASA Turbofan Jet Engine dataset (CMAPSS).  
This repo follows our course guidelines and is structured for collaboration (3 members).

## 🚀 Objectives
- Descriptive analysis & preprocessing
- Problem formalization (regression for RUL)
- Baseline models (Linear Regression, RandomForest)
- (Next) Sequence models (LSTM/GRU)
- Clear evaluation, plots, and report deliverables

## 📂 Repository Structure
```
.
├── data/                 # <keep empty in git> raw / interim / processed
├── docs/                 # project plan, guidelines, references (PPT/PDF)
├── notebooks/            # Jupyter notebooks (.ipynb)
├── reports/              # figures and final report exports
├── scripts/              # runnable scripts (download, training, eval)
├── src/                  # project modules (data, features, models, viz)
├── tests/                # smoke tests / unit tests
└── slides/               # course PPTs and final presentation
```
> **Note**: Add dataset files to `data/raw/` (e.g., `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).

## 🧑‍🤝‍🧑 Team & Roles
- **Owner:** Mullainathan V H  
- **Collaborators:** Teammate-1, Teammate-2  
- Suggested split:
  - *Data & EDA*: outliers, drift, correlation, feature selection
  - *Modeling*: baseline, tuning, advanced models (LSTM)
  - *MLOps & Reporting*: repo hygiene, CI, notebooks, final report/video

## 🛠️ Environment
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m ipykernel install --user --name turbofan-ml
```

## ▶️ Quickstart
```bash
# 0) Put NASA data into data/raw/
# 1) Run baseline training
python scripts/run_baseline.py --subset FD001

# 2) Open notebooks
jupyter lab
```

## 📈 Deliverables
- Notebooks (`.ipynb`), code, and plots
- Report PDF (in English), 4–5 min video
- GitHub repo (public / private with teacher access)

## 📜 License
MIT
