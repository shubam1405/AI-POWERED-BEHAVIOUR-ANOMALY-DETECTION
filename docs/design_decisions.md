# System Design

The project is designed using an object-oriented approach where each enterprise entity is represented as an independent model.

The `Company` class acts as the central source of truth and maintains relationships between departments, employees, devices, resources, sessions, and events.

Each module has a single responsibility:

- **Models** define enterprise entities.
- **Generators** create realistic synthetic data.
- **Simulator** manages sessions, events, and virtual time.
- **Validators** ensure data integrity and consistency.
- **Exporters** convert in-memory objects into datasets.
- **Machine Learning** detects behavioral anomalies.
- **Explainability** interprets model predictions.
- **Dashboard** visualizes enterprise activity and detected threats.

Behavior is modeled using probabilistic user profiles that define working hours, resource usage, browser preferences, login patterns, and session characteristics. These profiles are later used to generate realistic employee activity for anomaly detection.

The design emphasizes modularity, extensibility, and maintainability, making it easy to introduce new attack types, behavioral models, machine learning algorithms, and visualization components without affecting the overall architecture.