# 🛡️ Cyber Cage — AI-Powered Behavioral Anomaly Detection

> **Cyber Cage** is a full-stack **User & Entity Behavior Analytics (UEBA)** platform that detects insider threats and cyberattacks by learning what **"normal"** behavior looks like for every employee in an organization, then identifying, explaining, and prioritizing anomalous sessions for security analysts.

Since real enterprise security logs are highly sensitive and rarely labeled, Cyber Cage creates a realistic **synthetic enterprise environment** consisting of employees, devices, departments, working hours, and daily activities. It then injects **15 real-world cyberattack scenarios** into the generated data, enabling end-to-end training and evaluation of anomaly detection and attack classification models.

---

# 🚀 Features

### 🏢 Synthetic Enterprise Simulation

- Simulates a realistic enterprise with employees across **8 organizational roles**
  - System Admin
  - Software Engineer
  - HR
  - Finance
  - Sales
  - Security Analyst
  - Database Administrator
  - Network Engineer

- Generates role-specific behavioral profiles including:
  - Working hours
  - Device usage
  - Application access
  - Network behavior
  - Resource permissions

---
---

### 🔄 Universal Dataset Adapter

Cyber Cage is designed to work with both its **synthetic enterprise environment** and **external enterprise security datasets**.

A built-in **Universal Dataset Adapter** automatically converts heterogeneous security log formats into Cyber Cage's internal behavioural schema without requiring any modifications to the machine learning pipeline.

### Capabilities

- Automatic schema detection
- Flexible column mapping
- Timestamp normalization
- Event type normalization
- Internal event representation
- Smart session reconstruction
- Dataset compatibility validation
- Feature compatibility validation
- Confidence scoring
- Graceful handling of missing fields

### Supported Input Formats

- CSV
- JSON
- Pandas DataFrames

This enables Cyber Cage to ingest third-party enterprise datasets while preserving the existing GRU → XGBoost → SHAP → Copilot pipeline.
### 📊 Realistic User Activity Generation

Creates complete employee sessions containing events such as:

- Authentication
- File operations
- Process execution
- Network activity
- Email usage
- Application usage

Example workflow:

```text
Login
   ↓
Email
   ↓
IDE Usage
   ↓
Lunch Break
   ↓
Git Commits
   ↓
Logout
```

All events are generated with realistic timestamps and sequencing.

---

### ⚠️ Attack Simulation

Cyber Cage injects realistic attack behaviors into a subset of sessions across **15 attack categories**.

| Attack Categories |
|-------------------|
| Insider Threat |
| Credential Theft |
| Brute Force |
| Credential Stuffing |
| Impossible Travel |
| Privilege Escalation |
| Data Exfiltration |
| Lateral Movement |
| PowerShell Abuse |
| Malware Execution |
| Suspicious Process Execution |
| Device Spoofing |
| USB Data Theft |
| Off-hours Access |
| Unusual File Access |

---

### 🧠 Feature Engineering

Cyber Cage extracts **69 engineered behavioural features** from each reconstructed user session.

These features include:

- Authentication behaviour
- Device usage
- Network activity
- File operations
- Resource access
- Temporal behaviour
- Process execution
- Session statistics

The GRU Autoencoder then produces:

- Reconstruction Error
- Anomaly Score

These outputs are appended to create the final **71-feature vector**, which is used by the XGBoost attack classifier.

### 🔍 Anomaly Detection

Uses a **GRU/LSTM Autoencoder** trained exclusively on normal user behavior.

- Learns normal behavioral patterns
- Computes reconstruction error
- Flags sessions with unusually high reconstruction errors as anomalies

---

### 🎯 Attack Classification

A tuned **XGBoost** classifier predicts the specific attack type.

- **16 total classes**
  - 15 attack categories
  - Normal behavior

---

### 📈 Explainable AI (XAI)

Every prediction is explained using **SHAP**.

Provides:

- Global feature importance
- Per-class feature importance
- Local explanations for every session

---

### 🤖 AI Security Copilot

LLM-powered assistant capable of:

- Incident report generation
- Security recommendations
- Campaign correlation
- Interactive analyst Q&A

---

### 📊 Interactive SOC Dashboard

Built using **React**, **Tailwind CSS**, and **FastAPI**.

The dashboard provides:

- Live attack simulation
- Session analytics
- Behaviour anomaly detection
- Attack classification
- SHAP explainability
- Campaign correlation
- Threat timeline
- Incident reports
- AI Security Copilot
- Model Evaluation Dashboard
- Pipeline health monitoring
- Performance analytics
- ---

### 📈 Model Evaluation & Analytics

Cyber Cage includes a dedicated analytics dashboard for evaluating model performance and system health.

Features include:

- GRU ROC Curve
- Precision–Recall Curve
- Interactive Confusion Matrix
- Model performance metrics
- Enterprise dataset statistics
- Pipeline latency analysis
- Dynamic confidence indicators
- Circular risk gauge
- SHAP evidence visualization
- System health monitoring
- Evaluation report export

# 🏗️ System Architecture

```text
External Enterprise Dataset
        │
        │
        ├──────────────────────┐
        │                      │
Synthetic Enterprise           │
Generator                      │
        │                      │
        └──────────────┬───────┘
                       ▼
        Universal Dataset Adapter
                       │
      ┌────────────────────────────────┐
      │ Schema Detection               │
      │ Column Mapping                 │
      │ Timestamp Normalization        │
      │ Event Normalization            │
      │ Session Reconstruction         │
      └────────────────────────────────┘
                       │
                       ▼
       Feature Engineering (69 Features)
                       │
                       ▼
            GRU Autoencoder
                       │
      Reconstruction Error
         + Anomaly Score
                       │
                       ▼
           71-Feature Vector
                       │
                       ▼
      XGBoost Attack Classification
                       │
                       ▼
         SHAP Explainability
                       │
                       ▼
        AI Security Copilot
                       │
                       ▼
     React Dashboard (FastAPI)
```

---
# 🌐 External Dataset Compatibility

Cyber Cage is **dataset-agnostic**.

Alongside its synthetic enterprise generator, it supports ingestion of external enterprise security logs using the Universal Dataset Adapter.

The adapter automatically:

- Detects dataset schema
- Maps heterogeneous column names
- Normalizes timestamps
- Standardizes event types
- Reconstructs behavioural sessions
- Validates dataset compatibility
- Produces Cyber Cage native sessions

Workflow:

```text
External Dataset
        │
        ▼
Universal Dataset Adapter
        │
        ▼
Cyber Cage Session Schema
        │
        ▼
Feature Engineering
        │
        ▼
GRU Autoencoder
        │
        ▼
XGBoost Classifier
        │
        ▼
SHAP Explainability
        │
        ▼
AI Security Copilot
```

If the uploaded dataset lacks sufficient behavioural information, Cyber Cage generates a compatibility report highlighting missing fields and whether inference can proceed successfully.
# 📊 Model Performance

Evaluation performed on **1,953 held-out sessions**.

| Model | Metric | Score |
|--------|---------|------:|
| GRU Autoencoder | ROC-AUC | **0.82** |
| GRU Autoencoder | F1 Score (Anomaly) | **0.46** |
| XGBoost | Accuracy | **0.71** |
| XGBoost | ROC-AUC (OvR) | **0.97** |
| XGBoost | Macro Recall | **0.76** |

---

# 📂 Project Structure

```text
Cyber-Cage/
│
├── README.md
├── LICENSE
├── requirements.txt
├── package.json
├── .gitignore
├── main.py
├── test.py
├── dataset_adapter.py                  # Universal Dataset Adapter CLI
│
├── config/
│   ├── settings.yaml
│   ├── attack_profiles.yaml
│   ├── organization.yaml
│   ├── feature_config.yaml
│   ├── dataset_mapping.json
│   └── logging.yaml
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── system_design.md
│   ├── model_evaluation.md
│   ├── dataset_adapter.md
│   ├── dashboard.md
│   ├── assumptions.md
│   ├── experimental_results.md
│   └── design_decisions.md
│
├── models/
│   ├── gru_autoencoder.pt
│   ├── xgboost_attack_classifier.pkl
│   ├── shap_explainer.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── datasets/
│   ├── reports/
│   ├── explanations/
│   ├── metrics/
│   ├── simulations/
│   └── logs/
│
├── src/
│   │
│   ├── config/
│   │   ├── constants.py
│   │   ├── paths.py
│   │   └── settings.py
│   │
│   ├── models/
│   │   ├── employee.py
│   │   ├── session.py
│   │   ├── event.py
│   │   ├── device.py
│   │   ├── department.py
│   │   ├── enums.py
│   │   └── attack_types.py
│   │
│   ├── generators/
│   │   ├── enterprise_generator.py
│   │   ├── employee_generator.py
│   │   ├── device_generator.py
│   │   ├── event_generator.py
│   │   ├── session_generator.py
│   │   └── activity_generator.py
│   │
│   ├── simulator/
│   │   ├── attack_simulator.py
│   │   ├── combined_simulator.py
│   │   ├── campaign_simulator.py
│   │   ├── timeline_generator.py
│   │   └── scenario_builder.py
│   │
│   ├── features/
│   │   ├── feature_engineering.py
│   │   ├── sequence_builder.py
│   │   ├── dataset_adapter.py
│   │   ├── schema_detector.py
│   │   ├── event_normalizer.py
│   │   ├── session_builder.py
│   │   ├── feature_validator.py
│   │   └── compatibility_report.py
│   │
│   ├── behavior/
│   │   ├── behavior_knowledge_base.py
│   │   ├── cold_start_engine.py
│   │   ├── drift_monitor.py
│   │   └── profile_manager.py
│   │
│   ├── anomaly_detection/
│   │   ├── gru_autoencoder.py
│   │   ├── train_gru.py
│   │   ├── inference.py
│   │   └── thresholding.py
│   │
│   ├── attack_classification/
│   │   ├── train_xgboost.py
│   │   ├── inference.py
│   │   ├── dataset_loader.py
│   │   └── metrics.py
│   │
│   ├── explainability/
│   │   ├── shap_engine.py
│   │   ├── global_explanations.py
│   │   ├── local_explanations.py
│   │   └── explanation_report.py
│   │
│   ├── copilot/
│   │   ├── copilot_engine.py
│   │   ├── recommendation_engine.py
│   │   ├── incident_report.py
│   │   ├── campaign_analysis.py
│   │   ├── threat_summary.py
│   │   └── prompts.py
│   │
│   ├── dashboard/
│   │   ├── analytics.py
│   │   ├── metrics.py
│   │   ├── health_monitor.py
│   │   └── report_exporter.py
│   │
│   ├── api/
│   │   ├── routes.py
│   │   ├── simulation_api.py
│   │   ├── copilot_api.py
│   │   ├── analytics_api.py
│   │   └── health_api.py
│   │
│   ├── exporters/
│   │   ├── csv_exporter.py
│   │   ├── json_exporter.py
│   │   └── report_exporter.py
│   │
│   ├── validators/
│   │   ├── schema_validator.py
│   │   ├── feature_validator.py
│   │   └── data_validator.py
│   │
│   ├── utils/
│   │   ├── helpers.py
│   │   ├── logger.py
│   │   ├── timers.py
│   │   └── visualization.py
│   │
│   └── tests/
│       ├── test_dataset_adapter.py
│       ├── test_feature_engineering.py
│       ├── test_gru.py
│       ├── test_xgboost.py
│       ├── test_shap.py
│       ├── test_copilot.py
│       └── test_validation.py
│
├── frontend/
│   │
│   ├── public/
│   │
│   ├── src/
│   │   ├── assets/
│   │   ├── api/
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── styles/
│   │   │
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── SessionAnalytics.jsx
│   │   │   ├── ThreatTimeline.jsx
│   │   │   ├── CampaignGraph.jsx
│   │   │   ├── IncidentReport.jsx
│   │   │   ├── SecurityCopilot.jsx
│   │   │   ├── SHAPExplanation.jsx
│   │   │   ├── RiskGauge.jsx
│   │   │   ├── ConfidenceBadge.jsx
│   │   │   ├── SystemHealth.jsx
│   │   │   ├── ModelEvaluation.jsx
│   │   │   ├── PipelineLatency.jsx
│   │   │   ├── DatasetStatistics.jsx
│   │   │   ├── WhyFlagged.jsx
│   │   │   └── ExportReport.jsx
│   │   │
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   │
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   ├── external/
│   └── evaluation/
│
└── screenshots/
    ├── dashboard.png
    ├── analytics.png
    ├── copilot.png
    ├── campaign_graph.png
    ├── threat_timeline.png
    ├── model_evaluation.png
    └── dataset_adapter.png
```

### Directory Overview

| Directory | Description |
|------------|-------------|
| `config/` | Simulation configuration |
| `models/` | Core entity models and enums |
| `generators/` | Enterprise, employee, device, event generators |
| `simulator/` | Session scheduling and attack injection |
| `validators/` | Data integrity validation |
| `exporters/` | CSV and JSON export utilities |
| `features/` | Feature engineering and sequence building |
| `anomaly_detection/` | GRU/LSTM Autoencoder |
| `attack_classification/` | XGBoost classifier |
| `explainability/` | SHAP explanations |
| `copilot/` | LLM Security Copilot |
| `frontend/` | React dashboard |
| `outputs/` | Generated datasets and model outputs |
| `docs/` | Architecture and design documentation |

---

# ⚙️ Getting Started

## Prerequisites

- Python **3.9+**
- Node.js
- npm

---

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Generate Synthetic Enterprise Dataset

Generates:

- Enterprise
- Employees
- Sessions
- Events
- Attack scenarios
- Engineered features

```bash
python main.py
```

---

## 3️⃣ Train Models

### Train GRU/LSTM Autoencoder

```bash
python src/train_gru.py
```

### Train XGBoost Classifier

```bash
python src/train_xgboost.py
```

Both scripts support configurable hyperparameters via CLI.

```bash
python src/train_gru.py --help
python src/train_xgboost.py --help
```

---

## 4️⃣ Generate Explanations & Incident Reports

```bash
python src/explain_xgboost.py

python src/run_copilot.py --all
```

---

## 5️⃣ Run Backend & Dashboard

### Backend

```bash
python src/copilot_api.py
```

### Frontend

```bash
cd frontend

npm install

npm run dev
```

The FastAPI server exposes APIs for:

- Session analytics
- Dashboard feeds
- Incident reports
- Campaign correlation
- Analyst chat
- Live attack simulation

---

## 6️⃣ Run Tests

```bash
python -m unittest src/tests/test_validation.py
```

---

# 🛠️ Tech Stack

## Machine Learning

- PyTorch
- GRU/LSTM Autoencoder
- XGBoost
- SHAP
- scikit-learn

## Backend

- Python
- FastAPI

## Frontend

- React
- Vite
- Tailwind CSS
- Recharts

## Synthetic Data Generation

- Faker

---

# 📖 Documentation

For more detailed information, refer to:

```text
docs/
├── architecture.md
└── design_decisions.md
```

---

# 🎯 Key Highlights

- ✅ End-to-end UEBA platform
- ✅ Synthetic enterprise data generation
- ✅ 15 realistic cyberattack simulations
- ✅ 71 engineered behavioral features
- ✅ GRU/LSTM Autoencoder anomaly detection
- ✅ XGBoost multi-class attack classification
- ✅ SHAP explainability
- ✅ LLM-powered Security Copilot
- ✅ React + FastAPI dashboard
- ✅ Live SOC demonstration support

---

## ⭐ If you find this project useful, consider giving it a star!
