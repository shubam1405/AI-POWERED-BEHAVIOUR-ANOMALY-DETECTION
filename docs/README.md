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

- Extracts **71 behavioral features** per session
- Creates sequential representations for time-series modeling
- Includes behavioral statistics, temporal patterns, authentication behavior, device usage, file activity, and network behavior

---

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

### 📊 Interactive Dashboard

Built using **React** and served through **FastAPI**.

Includes:

- Live dashboards
- Session analytics
- Attack visualization
- Incident reports
- Campaign correlation
- Live simulation endpoints

---

# 🏗️ System Architecture

```text
Enterprise Generator
        │
        ▼
Session Generator
        │
        ▼
Event Generator
        │
        ▼
Attack Simulator
        │
        ▼
Feature Engineering (71 Features)
        │
        ▼
Sequence Builder
        │
        ▼
GRU/LSTM Autoencoder
        │
        ▼
Anomaly Detection
(Reconstruction Scores)
        │
        ▼
XGBoost Attack Classifier
        │
        ▼
Attack Type Prediction
        │
        ▼
SHAP Explainability
(Global + Local)
        │
        ▼
LLM Security Copilot
        │
        ▼
React Dashboard (FastAPI)
```

---

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
├── main.py
├── test.py
│
├── src/
│   ├── config/
│   ├── models/
│   ├── generators/
│   ├── simulator/
│   ├── validators/
│   ├── exporters/
│   ├── features/
│   ├── anomaly_detection/
│   ├── attack_classification/
│   ├── explainability/
│   ├── copilot/
│   ├── copilot_api.py
│   ├── train_gru.py
│   ├── train_xgboost.py
│   ├── run_copilot.py
│   └── tests/
│
├── frontend/
│
├── outputs/
│
└── docs/
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
