🛠️ NASA Turbofan Engine Remaining Useful Life (RUL) Prediction

This project focuses on predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.
It implements a complete machine learning pipeline including exploratory data analysis (EDA), feature engineering, classical machine learning models, and hyperparameter tuning.

The project follows a clean, reproducible ML workflow and is structured for academic evaluation and real-world scalability.

📌 Project Objectives

Understand degradation patterns in turbofan engines using sensor data

Perform structured EDA and feature engineering

Train and evaluate classical machine learning models

Prevent data leakage using engine-level cross-validation

Optimize model performance via hyperparameter tuning

Ensure reproducibility and clarity through modular code design


📂 Repository Structure
turbofan-ml-project-scaffold/
│
├── .github/                 # CI configuration
│
├── data/
│   ├── raw/                 # Original NASA CMAPSS dataset
│   ├── interim/             # Intermediate artifacts
│   └── processed/           # Preprocessed datasets (.parquet)
│
├── notebooks/               # Jupyter notebooks (EDA, preprocessing, models)
│   ├── 01_eda_baseline.ipynb
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_classical_models.ipynb
│   └── 03_hyperparameter_tuning.ipynb
│
├── scripts/                 # Script-based model execution
│   ├── run_baseline.py
│   ├── run_random_forest.py
│   └── run_lstm_torch.py
│
├── src/                     # Core source code
│   ├── data/                # Data loaders
│   ├── features/            # Feature engineering
│   ├── models/              # ML models
│   └── visualization/       # Plotting utilities
│
├── reports/                 # Saved metrics and trained models
├── slides/                  # Presentation materials
├── docs/                    # Project documentation
├── tests/                   # Basic tests
│
├── requirements.txt
├── README.md
└── LICENSE


📊 Dataset

This project uses the NASA CMAPSS Turbofan Engine Degradation Dataset, which contains multivariate time-series sensor data from simulated aircraft engines.

Dataset Subsets

FD001

FD002

FD003

FD004

Each subset represents different operating conditions and fault modes.

📎 Official source:
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

🛠️ Environment Setup
Prerequisites

Python 3.9+

VS Code / Jupyter Notebook

(Optional) Google Colab for hyperparameter tuning

Installation
git clone https://github.com/<your-username>/turbofan-ml-project-scaffold.git
cd turbofan-ml-project-scaffold


Create and activate a virtual environment (recommended):

python -m venv venv


Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

▶️ How to Run the Project
🔹 Step 1: Exploratory Data Analysis (Baseline)

📄 notebooks/01_eda_baseline.ipynb

Loads raw CMAPSS data

Performs initial EDA

Computes Remaining Useful Life (RUL)

Trains a simple baseline model

🔹 Step 2: EDA Preprocessing & Feature Engineering

📄 notebooks/01_eda_preprocessing.ipynb

Builds rolling and delta features

Handles missing values

Scales features

Saves processed datasets to data/processed/

🔹 Step 3: Classical Machine Learning Models

📄 notebooks/02_classical_models.ipynb

Trains Decision Tree, Random Forest, and SVR models

Uses GroupKFold to prevent engine-level data leakage

Evaluates models using MAE, RMSE, and R²

🔹 Step 4: Hyperparameter Tuning (Google Colab)

📄 notebooks/03_hyperparameter_tuning.ipynb

Performs RandomizedSearchCV

Optimizes model hyperparameters

Executed in Google Colab for faster computation

🔁 Recommended Execution Order
01_eda_baseline.ipynb
        ↓
01_eda_preprocessing.ipynb
        ↓
02_classical_models.ipynb
        ↓
03_hyperparameter_tuning.ipynb

▶️ Script-Based Execution (Optional)

Run models directly using scripts:

python scripts/run_baseline.py
python scripts/run_random_forest.py
python scripts/run_lstm_torch.py


Outputs are saved in the reports/ directory.

🧪 Reproducibility & Best Practices

Same dataset used across all stages

GroupKFold prevents data leakage

Fixed random seeds where applicable

Modular design for maintainability

📈 Results & Outputs

Evaluation metrics stored as JSON in reports/

Trained models saved for reuse

Visualizations available via notebooks

🧑‍⚖️ Notes for Evaluators

EDA and model development performed locally (VS Code)

Hyperparameter tuning executed in Google Colab for efficiency

Project follows a structured ML pipeline aligned with academic standards
