# Architecture

The AI-Powered Behavioral Anomaly Detection System follows a modular, layered architecture designed for scalability, maintainability, and realistic enterprise simulation.

The project is divided into independent modules responsible for configuration, enterprise modeling, data generation, validation, exporting, machine learning, and explainability.

The system first generates a realistic virtual enterprise consisting of departments, employees, devices, and enterprise resources. This synthetic organization acts as the foundation for generating behavioral sessions and security events.

Generated events are later processed by an LSTM Autoencoder to learn normal user behavior. Reconstruction errors are converted into anomaly scores, which are further analyzed using SHAP for explainability and summarized through an LLM-powered Security Copilot.

## High-Level Architecture

Enterprise Generator
→ Session Generator
→ Event Generator
→ Attack Simulator
→ Dataset Export
→ LSTM Autoencoder
→ Risk Scoring
→ SHAP Explainability
→ LLM Security Copilot
→ Dashboard

The modular design allows each component to be developed, tested, and extended independently while maintaining clear separation of responsibilities.