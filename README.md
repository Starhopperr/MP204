# Group G016 --- MP204 Cold Storage Spoilage (Kochi)

## Project Overview
This repository contains the end-to-end MLOps pipeline for **MP204 --- Cold Storage Spoilage (Kochi)** for the client **Kochi Cold Chain Cooperative**. The objective is to accurately predict `spoilage_pct` (regression measured by Mean Absolute Error, MAE) for agricultural lots stored across Chambers 1, 2, and 3 to mitigate spoilage and enable proactive sales.

## Group & Problem Details
- **Group ID**: `G016`
- **Problem ID**: `MP204`
- **Client**: Kochi Cold Chain Cooperative
- **Faculty**: Amit Soni | **Batch**: 52

### Team Members & Roles
| Enrolment No. | Name | Role | Core Responsibilities |
|---|---|---|---|
| **S24CSEU0994** | Mainak Sarkar | Repo Manager & Ops Lead | Repository layout, pinned requirements, params.yaml, GitHub Actions CI, Kubernetes configuration, Retraining & Monitoring orchestration |
| **S24CSEU1029** | Kunsh Kakkar | Data Lead | DVC setup, 7 data quality checks, data cleaning pipeline (`src/mp204/clean.py`), `findings.yaml` data issues |
| **S24CSEU1558** | Riya Yadav | Model Lead | Model training, SQLite MLflow tracking (`data_sha256`, MAE logging), Model Registry (`@champion` alias), Holdout predictions (`predictions/holdout_predictions.csv`), Model persistence |
| **S24CSEU0975** | Ahmed Bin Asad | Service & Ops Lead | FastAPI prediction service (`POST /predict`, `GET /health`), Pydantic validation rules, structured JSON logging, pytest suite, Docker containerization |

---

## Repository Structure
```
.
├── .github/workflows/ci.yml       # GitHub Actions workflow running pytest on push
├── .dvc/                          # DVC configuration
├── app/
│   ├── __init__.py
│   └── main.py                    # FastAPI application with /predict and /health
├── data/
│   ├── raw/                       # Raw datasets tracked by DVC (.dvc files)
│   │   ├── train.csv.dvc
│   │   ├── later.csv.dvc
│   │   └── holdout_features.csv
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment (mp204-api, 2 replicas)
│   └── service.yaml               # Kubernetes service (mp204-svc, port 80 -> 8000)
├── models/
│   └── model.joblib               # Serialized champion model
├── predictions/
│   └── holdout_predictions.csv    # Final holdout predictions (record_id,prediction)
├── reports/
│   └── findings.yaml              # Project findings and monitoring audit
├── src/
│   └── mp204/
│       ├── __init__.py
│       ├── clean.py               # Data cleaning and issue remediation
│       ├── data_checks.py         # Verification of 7 data quality issues
│       ├── monitor.py             # PSI drift monitoring and retrain logic
│       ├── train.py               # MLflow training and model registration
│       └── utils.py               # Config and parameter loader
├── tests/
│   ├── test_api.py                # FastAPI endpoints & Pydantic rules test
│   ├── test_data.py               # Data cleaning verification tests
│   └── test_model.py              # Model inference & pipeline tests
├── Dockerfile                     # Container definition (python:3.12-slim)
├── .dockerignore
├── findings.yaml                  # Findings form (marking tool target)
├── mlflow.db                      # Committed SQLite MLflow database
├── params.yaml                    # Pipeline parameters read via yaml.safe_load
├── requirements.txt               # Fully pinned dependencies
└── README.md
```

---

## Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Data Cleaning & Checks
```bash
python -m src.mp204.data_checks
python -m src.mp204.clean
```

### 3. Model Training & MLflow Registry
```bash
python -m src.mp204.train
```

### 4. Monitoring, Retraining & Holdout Predictions
```bash
python -m src.mp204.monitor
```

### 5. Run Tests
```bash
pytest
```

### 6. Run API Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
