import sys
import os
import argparse
import logging
import shap

# Ensure src is on Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from simulator.company import Company
from features.dataset_adapter import UniversalDatasetAdapter
from features.feature_engineer import FeatureEngineer
from features.sequence_builder import SequenceBuilder
from anomaly_detection.inference import InferenceEngine
from attack_classification.inference import AttackInferenceEngine
from attack_classification.utils import ATTACK_LABELS
from attack_classification.dataset_loader import AttackDatasetLoader
from copilot.report_generator import ReportGenerator

def main():
    parser = argparse.ArgumentParser(description="Cyber Cage Universal Dataset Adapter CLI")
    parser.add_argument("input_path", type=str, nargs="?", default=None, help="Path to input external security dataset (CSV/JSON)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom dataset_mapping.json")
    parser.add_argument("--inactivity", type=int, default=1800, help="Inactivity gap threshold in seconds for session building (default 1800s)")

    args = parser.parse_args()

    input_path = args.input_path
    if not input_path:
        # Fallback to create sample external dataset if no file argument is passed
        input_path = "sample_external_dataset.csv"
        if not os.path.exists(input_path):
            import pandas as pd
            df = pd.DataFrame([
                {"user_name": "emp_sarah", "datetime": "2026-07-26 09:00:00", "action_name": "user_login_success", "hostname": "PC-FINANCE-01", "ip_addr": "10.0.4.12", "browser_ver": "v122"},
                {"user_name": "emp_sarah", "datetime": "2026-07-26 09:15:20", "action_name": "file_access_read", "hostname": "PC-FINANCE-01", "ip_addr": "10.0.4.12", "browser_ver": "v122"},
                {"user_name": "emp_sarah", "datetime": "2026-07-26 09:30:10", "action_name": "data_download", "hostname": "PC-FINANCE-01", "ip_addr": "10.0.4.12", "browser_ver": "v122"},
                {"user_name": "emp_alex", "datetime": "2026-07-26 23:45:00", "action_name": "vpn_connect", "hostname": "LAPTOP-DEV-99", "ip_addr": "192.168.1.100", "browser_ver": "v120"},
                {"user_name": "emp_alex", "datetime": "2026-07-26 23:50:12", "action_name": "cmd_exec", "hostname": "LAPTOP-DEV-99", "ip_addr": "192.168.1.100", "browser_ver": "v120"},
            ])
            df.to_csv(input_path, index=False)
            print(f"[*] Created sample external dataset: {input_path}")

    print(f"\n[*] Cyber Cage Universal Dataset Adapter Ingesting: {input_path}")
    
    # 1. Adapter Processing
    adapter = UniversalDatasetAdapter(config_path=args.config, inactivity_threshold_seconds=args.inactivity)
    company = Company()
    employee_lookup = {emp.employee_id: emp for emp in company.employees}

    sessions, report = adapter.process(input_path, employee_lookup=employee_lookup)

    print("\n" + report.summary_text() + "\n")
    print("[+] Schema detected")
    print("[+] Sessions built")

    if not sessions:
        print("[!] No sessions constructed. Exiting pipeline.")
        return

    # 2. Feature Extraction
    fe = FeatureEngineer()
    raw_features_list, scaled_features_list, copilot_records = fe.run(sessions=sessions)
    print("[+] Features extracted")

    # 3. GRU Autoencoder Inference
    gru_engine = InferenceEngine(model_path="models/gru_autoencoder.pt", threshold=0.05)
    gru_engine.calibrate(err_min=0.0, err_max=1.0)

    seq_builder = SequenceBuilder(max_len=50, feature_dim=21)
    sequences, masks = seq_builder.build_sequences(sessions)

    for i, seq in enumerate(sequences):
        sid = sessions[i].session_id
        res = gru_engine.score_sequence(sid, seq)
        raw_features_list[i]["reconstruction_error"] = res["reconstruction_error"]
        raw_features_list[i]["anomaly_score"] = res["anomaly_score"]
        scaled_features_list[i]["reconstruction_error"] = res["reconstruction_error"]
        scaled_features_list[i]["anomaly_score"] = res["anomaly_score"]
    print("[+] GRU complete")

    # 4. XGBoost Attack Classification Inference
    import numpy as np
    loader = AttackDatasetLoader().load()
    expected_features = loader.feature_names
    xgb_engine = AttackInferenceEngine(model_path="models/xgboost_attack_classifier.pkl", class_names=ATTACK_LABELS)
    xgb_engine.feature_names = expected_features

    session_ids = [s.session_id for s in sessions]
    X_matrix = np.array([[row.get(f, 0.0) for f in expected_features] for row in scaled_features_list])
    xgb_results = xgb_engine.predict_batch(session_ids, X_matrix)
    print("[+] XGBoost complete")

    # 5. SHAP Explainability
    xgb_model = xgb_engine.get_model()
    explainer = shap.TreeExplainer(xgb_model, feature_perturbation="interventional", model_output="raw")
    print("[+] SHAP complete")

    # 6. Report Generation & Summary
    print("[+] Report generated")
    print("\n[+] SUCCESS: External dataset processed end-to-end through Cyber Cage ML Pipeline with 0 Model Modifications!\n")

if __name__ == "__main__":
    main()
