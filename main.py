import os
import sys
import random
from datetime import datetime

# Add src/ to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "src",
        )
    ),
)

from config.config import SIMULATION_CONFIG, RANDOM_SEED
from exporters.csv_exporter import CSVExporter
from exporters.sessions_exporter import SessionsExporter
from generators.enterprise_generator import EnterpriseGenerator
from simulator.clock import SimulationClock
from simulator.session_scheduler import SessionScheduler
from simulator.session_generator import SessionGenerator
from validators.session_validator import SessionValidator
from validators.event_validator import EventValidator
from validators.anomaly_validator import AnomalyValidator
from utils.validator import CompanyValidator, ValidationError
from utils.id_generator import IdGenerator
from utils.logger import setup_logger
from models.enums import LoginMethod
from generators.event_generator import EventGenerator
from exporters.events_exporter import EventsExporter
from simulator.anomaly_injector import AnomalyInjector
from exporters.anomaly_exporter import AnomalyExporter
from features.feature_engineer import FeatureEngineer
from features.feature_validator import FeatureValidator
from exporters.feature_exporter import FeatureExporter
from features.sequence_builder import SequenceBuilder


def main() -> None:
    logger = setup_logger("Main")

    logger.info("=" * 60)
    logger.info("Enterprise Behavioral Simulator - Phase 2A")
    logger.info("=" * 60)

    try:
        # Initialize simulation clock
        start_time = datetime(2026, 6, 1, 0, 0, 0)
        clock = SimulationClock(start_time)
        logger.info(f"Simulation Clock initialized at: {clock}")

        # 1. Generate Virtual Enterprise
        logger.info("Generating enterprise structure...")
        company = EnterpriseGenerator().generate()
        logger.info("Enterprise generation completed.")

        # 2. Validate Company structure
        logger.info("Running Company validation...")
        CompanyValidator().validate(company)
        logger.info("Company validation successful.")

        # 3. Export Company CSVs
        logger.info("Exporting company CSV files...")
        company_exporter = CSVExporter(company, output_dir="data/raw")
        company_exporter.export_all()
        logger.info("Company CSV export completed.")

        # 4. Initialize random generator with seed
        rng = random.Random(RANDOM_SEED)

        # 5. Session Scheduling
        simulation_days = SIMULATION_CONFIG.get("simulation_days", 30)
        logger.info(f"Scheduling sessions for {simulation_days} simulation days...")
        scheduler = SessionScheduler(company, clock.get_time(), simulation_days, rng)
        scheduled_slots = scheduler.schedule_sessions()

        # 6. Session Generation
        logger.info("Generating session details...")
        session_id_gen = IdGenerator(prefix="SES-", start=1)
        generator = SessionGenerator(company, session_id_gen, rng)
        sessions = generator.generate_sessions(scheduled_slots)
        logger.info(f"Generated {len(sessions)} sessions.")

        # 7. Session Validation
        logger.info("Running session validation...")
        SessionValidator().validate(sessions, company)
        logger.info("Session validation successful.")

        # 8. Export Sessions to CSV
        logger.info("Exporting sessions to CSV...")
        sessions_exporter = SessionsExporter(sessions, output_dir="data/raw")
        sessions_exporter.export()
        logger.info("Sessions CSV export completed.")

        # 9. Event Generation
        logger.info("Generating session events...")
        event_gen = EventGenerator(company, rng)
        events = event_gen.generate_events(sessions)
        logger.info(f"Generated {len(events)} events.")

        # 10. Event Validation
        logger.info("Running event validation...")
        EventValidator().validate(sessions)
        logger.info("Event validation successful.")

        # 11. Export Events to CSV
        logger.info("Exporting events to CSV...")
        events_exporter = EventsExporter(events, output_dir="data/raw")
        events_exporter.export()
        logger.info("Events CSV export completed.")

        # 12. Anomaly Injection
        logger.info("Injecting anomalies...")
        injector = AnomalyInjector(company, rng)
        anom_sessions, anom_events = injector.inject_anomalies(sessions)
        logger.info(f"Anomaly injection completed. {sum(1 for s in anom_sessions if s.is_anomalous)} sessions modified.")

        # 13. Anomaly Validation
        logger.info("Running anomaly validation...")
        AnomalyValidator().validate(anom_sessions, company)
        logger.info("Anomaly validation successful.")

        # 14. Export anomalous data to CSV
        logger.info("Exporting anomalous data...")
        anomaly_exporter = AnomalyExporter(anom_sessions, anom_events, output_dir="data/raw")
        anomaly_exporter.export()
        logger.info("Anomalous CSV export completed.")

        # 15. Pass anomalous sessions directly to FeatureEngineer
        # ----------------------------------------------------------------
        # KEY FIX: We skip DataAdapter re-sessionization here.  The
        # anom_sessions objects already carry is_anomalous, attack_type,
        # and risk_score set by AnomalyInjector.  Re-reading from CSV
        # would reconstruct bare Session objects and lose all that metadata.
        # ----------------------------------------------------------------
        feat_engineer = FeatureEngineer(data_dir="data/raw", company=company)

        # --- Pre-FE validation logging -----------------------------------
        anomalous_preview = [s for s in anom_sessions if s.is_anomalous][:3]
        if anomalous_preview:
            logger.info("Pre-FE validation — sample anomalous sessions:")
            for s in anomalous_preview:
                logger.info(
                    f"  {s.session_id}  is_anomalous={s.is_anomalous}  "
                    f"attack_type={getattr(s.attack_type, 'value', s.attack_type)}  "
                    f"risk_score={s.risk_score:.1f}"
                )
        else:
            logger.warning("Pre-FE validation — no anomalous sessions found after injection!")

        # Extract features from the in-memory anomalous sessions
        logger.info("Extracting session feature vectors...")
        raw_features, scaled_features, copilot_context = feat_engineer.run(sessions=anom_sessions)
        logger.info(f"Feature engineering completed. Processed {len(raw_features)} feature vectors.")

        # 16. Feature Validation
        logger.info("Running feature validation...")
        FeatureValidator().validate(raw_features, len(anom_sessions))
        logger.info("Feature validation successful.")

        # 17. Export Tabular Features to CSV
        logger.info("Exporting tabular features to CSV...")
        feat_exporter = FeatureExporter(raw_features, scaled_features, output_dir="data/processed")
        feat_exporter.export()
        logger.info("Tabular features CSV export completed.")

        # 18. Sequence Building (Sequential Features for GRU Autoencoder)
        logger.info("Building sequential features for GRU Autoencoder...")
        seq_builder = SequenceBuilder(max_len=50)
        sequences, masks = seq_builder.build_sequences(anom_sessions)
        seq_builder.save(sequences, masks, output_dir="data/processed")
        logger.info("Sequential features export completed.")

        # 19. Verify Generated Files
        logger.info("Verifying output files:")
        raw_files = ["employees.csv", "devices.csv", "resources.csv", "sessions.csv", "events.csv", "sessions_anomalous.csv", "events_anomalous.csv"]
        proc_files = ["tabular_features.csv", "tabular_features_scaled.csv", "scaler.pkl", "sequential_features.json", "copilot_context.json", "feature_metadata.json"]
        for filename in raw_files:
            filepath = os.path.join("data/raw", filename)
            if os.path.exists(filepath):
                logger.info(f"[OK] {filename:<25} {os.path.getsize(filepath)} bytes")
            else:
                logger.warning(f"[FAIL] Missing {filename}")
        for filename in proc_files:
            filepath = os.path.join("data/processed", filename)
            if os.path.exists(filepath):
                logger.info(f"[OK] processed/{filename:<22} {os.path.getsize(filepath)} bytes")
            else:
                logger.warning(f"[FAIL] Missing processed/{filename}")

        # 20. Summary Report
        logger.info("=" * 60)
        logger.info("Simulation Execution Summary")
        logger.info("=" * 60)
        
        total_sessions = len(sessions)
        total_events = len(events)
        num_employees = len(company.employees)
        avg_sessions_per_emp = total_sessions / num_employees if num_employees > 0 else 0
        avg_events_per_sess = total_events / total_sessions if total_sessions > 0 else 0
        
        total_duration = sum(s.duration_seconds for s in sessions)
        avg_duration_minutes = (total_duration / total_sessions / 60) if total_sessions > 0 else 0
        
        office_sessions = sum(1 for s in sessions if s.login_method == LoginMethod.OFFICE)
        vpn_sessions = sum(1 for s in sessions if s.login_method == LoginMethod.VPN)

        logger.info(f"Simulation Period      : {simulation_days} Days")
        logger.info(f"Total Employees        : {num_employees}")
        logger.info(f"Total Sessions         : {total_sessions}")
        logger.info(f"Avg Sessions / Employee: {avg_sessions_per_emp:.2f}")
        logger.info(f"Avg Session Duration   : {avg_duration_minutes:.2f} minutes")
        logger.info(f"Office Login Sessions  : {office_sessions} ({office_sessions/total_sessions*100:.1f}%)" if total_sessions > 0 else "Office Login Sessions  : 0")
        logger.info(f"VPN Login Sessions     : {vpn_sessions} ({vpn_sessions/total_sessions*100:.1f}%)" if total_sessions > 0 else "VPN Login Sessions     : 0")
        logger.info(f"Total Events Generated : {total_events}")
        logger.info(f"Avg Events / Session   : {avg_events_per_sess:.2f}")
        
        # Anomalous stats
        num_anom_sessions = sum(1 for s in anom_sessions if s.is_anomalous)
        num_anom_events = sum(1 for e in anom_events if e.is_anomalous)
        logger.info(f"Anomalous Sessions     : {num_anom_sessions} ({num_anom_sessions/total_sessions*100:.1f}%)" if total_sessions > 0 else "Anomalous Sessions     : 0")
        logger.info(f"Anomalous Events       : {num_anom_events} ({num_anom_events/total_events*100:.1f}%)" if total_events > 0 else "Anomalous Events       : 0")
        
        # Feature stats
        logger.info(f"Total Tabular Columns  : {len(raw_features[0]) - 5}")
        logger.info(f"Sequential Tensor Shape: ({len(anom_sessions)}, 50, 21)")
        logger.info(f"Intelligence Fusion    : Risk, SHAP Mapping, Copilot Context exported")
        logger.info("=" * 60)
        logger.info("Phase 4 execution completed successfully.")

    except ValidationError as ve:
        logger.error(f"Company Validation failed:\n{ve}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()