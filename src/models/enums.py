from enum import Enum

class EventStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class BaseEnum(Enum):
    """
    Base Enum providing case-insensitive string conversion.
    """

    @classmethod
    def from_str(cls, value: str):
        value = value.strip().lower()
        for member in cls:
            if member.value.lower() == value:
                return member
        raise ValueError(f"Unknown {cls.__name__}: {value}")


# ============================================================
# Departments
# ============================================================

class DepartmentName(BaseEnum):
    FINANCE = "Finance"
    ENGINEERING = "Engineering"
    HR = "HR"
    SALES = "Sales"
    IT = "IT"
    MARKETING = "Marketing"


# ============================================================
# Device Types
# ============================================================

class DeviceType(BaseEnum):
    LAPTOP = "Laptop"
    DESKTOP = "Desktop"
    MOBILE = "Mobile"


# ============================================================
# Operating Systems
# ============================================================

class OperatingSystem(BaseEnum):
    WINDOWS = "Windows"
    LINUX = "Linux"
    MACOS = "macOS"
    IOS = "iOS"
    ANDROID = "Android"


# ============================================================
# Browsers
# ============================================================

class Browser(BaseEnum):
    CHROME = "Chrome"
    FIREFOX = "Firefox"
    EDGE = "Edge"
    SAFARI = "Safari"


# ============================================================
# Enterprise Resources
# ============================================================

class ResourceType(BaseEnum):
    EMAIL = "Email"
    SLACK = "Slack"
    GITHUB = "GitHub"
    FILE_SERVER = "FileServer"
    FINANCE_DB = "FinanceDB"
    HR_PORTAL = "HRPortal"
    PAYROLL = "Payroll"
    CRM = "CRM"
    ADMIN_CONSOLE = "AdminConsole"
    JIRA = "Jira"
    SERVERS = "Servers"
    VPN = "VPN"
    CONFLUENCE = "Confluence"
    TERMINAL = "Terminal"
    HRMS = "HRMS"
    EMPLOYEE_RECORDS = "EmployeeRecords"
    ERP = "ERP"
    ACCOUNTING = "Accounting"
    INVOICES = "Invoices"
    MEETINGS = "Meetings"
    ANALYTICS = "Analytics"
    CAMPAIGN_DASHBOARD = "CampaignDashboard"
    MONITORING = "Monitoring"
    FIREWALL = "Firewall"


# ============================================================
# User Access Levels
# ============================================================

class AccessLevel(BaseEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ============================================================
# Data Sensitivity
# ============================================================

class SensitivityLevel(BaseEnum):
    PUBLIC = "Public"
    INTERNAL = "Internal"
    CONFIDENTIAL = "Confidential"
    RESTRICTED = "Restricted"


# ============================================================
# Event Types
# ============================================================

class EventType(BaseEnum):
    LOGIN = "Login"
    LOGOUT = "Logout"
    RESOURCE_ACCESS = "ResourceAccess"
    FILE_READ = "FileRead"
    FILE_WRITE = "FileWrite"
    FILE_DOWNLOAD = "FileDownload"
    FILE_UPLOAD = "FileUpload"
    PROCESS_START = "ProcessStart"
    PROCESS_STOP = "ProcessStop"
    API_CALL = "ApiCall"
    DATABASE_QUERY = "DatabaseQuery"
    VPN_CONNECT = "VPNConnect"
    VPN_DISCONNECT = "VPNDisconnect"
    NETWORK_CONNECT = "NetworkConnect"
    NETWORK_DISCONNECT = "NetworkDisconnect"


# ============================================================
# Attack Types
# ============================================================

class AttackType(BaseEnum):
    NONE = "None"
    BRUTE_FORCE = "Brute Force"
    CREDENTIAL_STUFFING = "Credential Stuffing"
    CREDENTIAL_MISUSE = "Credential Misuse"
    IMPOSSIBLE_TRAVEL = "Impossible Travel"
    DEVICE_SPOOFING = "Device Spoofing"
    LATERAL_MOVEMENT = "Lateral Movement"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    DATA_EXFILTRATION = "Data Exfiltration"
    INSIDER_THREAT = "Insider Threat"
    OFF_HOURS_ACCESS = "Off-hours Access"
    USB_DATA_THEFT = "USB Data Theft"
    MALWARE_EXECUTION = "Malware Execution"
    SUSPICIOUS_POWERSHELL = "Suspicious PowerShell"
    BEACONING_C2 = "Beaconing C2"
    LOW_SLOW_EXFILTRATION = "Low-and-Slow Exfiltration"
    INSIDER_DRIFT = "Insider Drift"


# ============================================================
# Session Status
# ============================================================

class SessionStatus(BaseEnum):
    ACTIVE = "Active"
    COMPLETED = "Completed"
    TERMINATED = "Terminated"
    FAILED = "Failed"


# ============================================================
# Login Methods
# ============================================================

class LoginMethod(BaseEnum):
    OFFICE = "Office"
    VPN = "VPN"


# ============================================================
# Login Status
# ============================================================

class LoginStatus(BaseEnum):
    SUCCESS = "Success"
    FAILURE = "Failure"
    LOCKED = "Locked"


# ============================================================
# Risk Level
# ============================================================

class RiskLevel(BaseEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"