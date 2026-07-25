Cyber Cage — AI-Powered Behavioral Anomaly Detection

Cyber Cage is a full-stack User & Entity Behavior Analytics (UEBA) platform that detects insider threats and cyberattacks by learning what "normal" looks like for every employee in an organization — then flagging, explaining, and triaging the sessions that deviate from it.

Since real enterprise security logs are sensitive and rarely labeled, Cyber Cage generates its own realistic synthetic enterprise: employees, devices, departments, working hours, and day-to-day activity — then injects 15 categories of real-world attack behavior into that data to train and evaluate the detection pipeline end-to-end.

What it does
Simulates a virtual enterprise — employees across 8 roles (System Admin, Software Engineer, HR, Finance, Sales, Security Analyst, DBA, Network Engineer), each with role-specific behavioral profiles (working hours, devices, apps, access patterns).
Generates realistic activity sessions — authentication, file operations, process activity, network activity, email, and application usage, sequenced with realistic timing (e.g. login → email → IDE → lunch → commits → logout).
Injects attack scenarios into a subset of sessions across 15 categories: Insider Threat, Credential Theft, Brute Force, Credential Stuffing, Impossible Travel, Privilege Escalation, Data Exfiltration, Lateral Movement, PowerShell Abuse, Malware Execution, Suspicious Process Execution, Device Spoofing, USB Data Theft, Off-hours Access, and Unusual File Access.
Engineers 71 behavioral features per session and builds sequential representations for time-series modeling.
Detects anomalies with a GRU/LSTM Autoencoder trained on normal behavior — sessions with high reconstruction error are flagged as anomalous.
Classifies attack type with a tuned XGBoost model across 16 classes (15 attack types + Normal).
Explains every prediction using SHAP (global feature importance, per-class importance, and per-session local explanations).
Summarizes incidents with an LLM Security Copilot — generates human-readable incident reports, recommendations, campaign correlation across related sessions, and supports analyst Q&A chat.
Visualizes everything in a React dashboard served via a FastAPI backend, with live simulation endpoints for SOC-style demos.
Architecture
Enterprise Generator
      ↓
Session Generator → Event Generator → Attack Simulator
      ↓
Feature Engineering (71 features) → Sequence Builder
      ↓
GRU/LSTM Autoencoder  →  Anomaly / Reconstruction Scores
      ↓
XGBoost Attack Classifier  →  Attack Type Predictions
      ↓
SHAP Explainability  →  Global + Local Explanations
      ↓
LLM Security Copilot  →  Incident Reports, Recommendations, Campaign Correlation
      ↓
React Dashboard (via FastAPI)
Results

On the held-out evaluation set (1,953 sessions):

Model	Metric	Score
GRU Autoencoder (anomaly detection)	ROC-AUC	0.82
GRU Autoencoder (anomaly detection)	F1 (anomaly class)	0.46
XGBoost (16-class attack classification)	Accuracy	0.71
XGBoost (16-class attack classification)	ROC-AUC (OvR)	0.97
XGBoost (16-class attack classification)	Recall (macro)	0.76
Project structure
├── main.py                    # Generates enterprise, sessions, events, attacks, and features
├── test.py                    # Test runner
├── src/
│   ├── config/                # Simulation configuration
│   ├── models/                # Core entity dataclasses & enums (Employee, Device, Session, ...)
│   ├── generators/             # Employee / device / resource / enterprise / event generators
│   ├── simulator/              # Company state, virtual clock, session scheduling, attack injection
│   ├── validators/              # Data integrity checks
│   ├── exporters/               # CSV / JSON dataset export
│   ├── features/                # Feature engineering, scaling, sequence building, behavior knowledge base
│   ├── anomaly_detection/       # GRU/LSTM Autoencoder — training, inference, evaluation
│   ├── attack_classification/   # XGBoost classifier — training, inference, evaluation
│   ├── explainability/          # SHAP-based global & local explanations
│   ├── copilot/                 # LLM-powered incident summarization, recommendations, chat, correlation
│   ├── copilot_api.py           # FastAPI REST server (dashboard, reports, chat, simulation)
│   ├── train_gru.py             # CLI: train the anomaly detection model
│   ├── train_xgboost.py         # CLI: train the attack classifier
│   ├── run_copilot.py           # CLI: generate incident reports / chat with the copilot
│   └── tests/                   # Unit test suite
├── frontend/                   # React + Vite dashboard (Tailwind, Recharts)
├── outputs/                     # Generated datasets, model outputs, metrics, SHAP values, plots
└── docs/                        # Architecture & design documentation
Getting started
Prerequisites
Python >= 3.9
Node.js (for the dashboard)
1. Install dependencies
bash
pip install -r requirements.txt
2. Generate the synthetic enterprise dataset

Simulates the enterprise, schedules and generates sessions/events, injects attacks, and builds engineered features:

bash
python main.py
3. Train the models

Train the GRU/LSTM autoencoder for anomaly detection:

bash
python src/train_gru.py

Train the XGBoost attack classifier:

bash
python src/train_xgboost.py

Both scripts accept CLI flags for hyperparameters, output paths, and tuning — run with --help for the full list.

4. Generate explanations & incident reports
bash
python src/explain_xgboost.py
python src/run_copilot.py --all
5. Run the API + dashboard
bash
# Backend
python src/copilot_api.py

# Frontend (in a separate terminal)
cd frontend
npm install
npm run dev

The FastAPI server exposes endpoints for session data, incident reports, dashboard feeds, campaign correlation, analyst chat, and live attack simulation for demos.

6. Run tests
bash
python -m unittest src/tests/test_validation.py
Tech stack

ML/Backend: Python, PyTorch (GRU Autoencoder), XGBoost, SHAP, FastAPI, scikit-learn Frontend: React, Vite, Tailwind CSS, Recharts Data generation: Faker

Documentation

See docs/architecture.md and docs/design_decisions.md for a deeper dive into system design.
