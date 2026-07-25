# SOC Incident Report: SES-008220

## 1. Executive Summary
The Cyber Cage UEBA engine detected anomalous behaviors in session **SES-008220** associated with Employee **DEV-AGENT-0002**. The session was classified as **Brute Force** with a confidence score of **78.1%**, indicating a **High** severity risk. Immediate containment action is recommended.

## 2. Incident Overview
- **Session ID:** SES-008220
- **Employee ID:** DEV-AGENT-0002
- **Source IP Address:** 10.10.243.241
- **Device ID:** DEV-0002
- **Hour of Day:** 7
- **Duration:** Unknown

## 3. Attack Classification & Prediction
- **Primary Prediction:** Brute Force
- **Classifier Confidence:** 78.1%
- **Top 3 Candidate Classes:** Brute Force (78%), Credential Stuffing (18%), Device Spoofing (1%)

## 4. Severity Assessment
The security risk level is determined to be **High**. This is based on a cumulative risk score of **36.10** and an isolated reconstruction reconstruction error of **0.9415** from the GRU autoencoder network.

## 5. MITRE ATT&CK Mapping
- **Tactic:** Credential Access
- **Technique:** T1110 - Brute Force
- **Mapping Context:** Mapped directly based on Credential Access (T1110 - Brute Force).

## 6. Behavioral Indicators
The session exhibited significant deviations from historical employee baselines. Key anomalous variables include:
- Total risk score: 36.1
- Anomaly score: 0.9415
- Mapped attack signature similarity: 78.1%

## 7. Key SHAP Evidence
The following behavioral features contributed most significantly to the anomaly detection:
- **processes_started**: Value of 0.0 introduced a low positive deviation.
- **process_execution_position**: Value of -1.0 introduced a negligible deviation.
- **maximum_idle_time**: Value of 164.0 introduced a negligible deviation.
- **process_event_ratio**: Value of 0.0 introduced a negligible deviation.

## 8. Exonerating/Counteracting Factors
The following features displayed baseline behaviors that counteracted the threat classification:
- **external_connections**: Value of 0.0 acted as a mitigating factor (high negative).
- **beaconing_event_count**: Value of 0.0 acted as a mitigating factor (high negative).
- **resource_event_ratio**: Value of 0.1429 acted as a mitigating factor (low negative).
- **processes_stopped**: Value of 0.0 acted as a mitigating factor (negligible).

## 9. Risk Assessment
With a risk score of **36.10**, this session represents a marked deviation from regular activity. The GRU network's anomaly index (**0.9415**) verifies that sequence reconstruction was highly corrupted, establishing objective evidence of behavioral drift.

## 10. Potential Business Impact
Credential compromise may allow unauthorised access to systems and sensitive data. If successful, attackers gain persistent access. Potential impact: data breach, compliance violation, account takeover.

## 11. Containment Actions
The security operations team must perform these immediate containment steps:
- [ ] Lock the targeted account(s) immediately\n- [ ] Block the source IP address at the perimeter firewall\n- [ ] Enable MFA on the targeted account if not already enforced\n- [ ] Alert the account owner via a secondary channel\n- [ ] Notify IT Security Manager immediately\n

## 12. Investigation Checklist
Forensic and triage investigative checklist:
- [ ] Determine whether any login attempts succeeded\n- [ ] Review all activity from the source IP across all accounts\n- [ ] Check the source IP against threat intelligence feeds\n- [ ] Identify whether multiple accounts were targeted simultaneously\n- [ ] Perform full directory access audit for employee account\n

## 13. Remediation Plan
Follow-up remediation steps to resolve root causes:
- [ ] Enforce account lockout policy after N failed attempts\n- [ ] Implement CAPTCHA or rate-limiting on the authentication endpoint\n- [ ] Enable MFA organisation-wide for all privileged accounts\n- [ ] Reset the targeted account's password\n

## 14. Recovery Recommendations
Systems recovery playbooks:
- [ ] Unlock account after credentials are confirmed reset\n- [ ] Monitor account for 7 days for follow-on anomalous activity\n- [ ] Review and update authentication policy\n

## 15. Lessons Learned
- Verify user awareness concerning appropriate access policies.
- Audit resource permission trees regularly to enforce least-privilege policies.
- Deploy additional telemetry triggers based on the top SHAP indicators identified in this incident.
