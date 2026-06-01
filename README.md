# Flight Envelope Prediction & Safe Operating Zone Classification for Defence eVTOLs

## Overview

This project presents a machine learning-based safety assessment system for Defence eVTOL (Electric Vertical Take-Off and Landing) aircraft. The system predicts whether an eVTOL is operating within a **Safe**, **Marginal**, or **Unsafe** flight envelope based on real-time environmental and vehicle-state parameters.

The objective is to help identify operational boundaries, improve mission planning, and enhance flight safety by learning from simulation-generated aerospace data.

Developed as part of **Project Urdhyuth** at the **NMCAD Lab, Department of Aerospace Engineering, Indian Institute of Science (IISc), Bengaluru**.

---

## Problem Statement

Defence eVTOLs operate under varying environmental and mission conditions where safety depends on multiple interacting factors such as:

- Wind Speed
- Altitude
- Payload
- Ambient Temperature
- Battery State of Charge (SOC)
- Battery Temperature
- Climb Rate
- Vertical Speed
- Thrust Demand

Determining whether a vehicle can safely complete a mission under these conditions is a complex aerospace engineering challenge.

This project formulates the problem as a multiclass classification task:

| Class ID | Classification |
|-----------|---------------|
| 0 | Marginal |
| 1 | Safe |
| 2 | Unsafe |

---

## Objectives

- Develop a machine learning model to classify eVTOL operating conditions.
- Predict Safe, Marginal, and Unsafe flight zones in real time.
- Improve operational safety and mission planning.
- Provide explainable insights into critical flight parameters.
- Demonstrate the applicability of AI in safety-critical aerospace systems.

---

## Dataset Generation

A physics-based simulation framework was used to generate thousands of flight scenarios.

### Input Parameters

- Wind Speed
- Altitude
- Payload Weight
- Ambient Temperature
- Battery State of Charge (SOC)
- Battery Temperature
- Vertical Speed
- Thrust Demand

### Safety Labeling

Each scenario was classified according to engineering constraints and operational limits:

- Maximum thrust availability
- Thermal safety thresholds
- Minimum battery reserve requirements
- Wind tolerance limits
- Flight envelope restrictions

Resulting labels:

- Safe
- Marginal
- Unsafe

---

## Machine Learning Pipeline

### 1. Data Preprocessing

- Data cleaning
- Feature normalization
- Handling invalid operating points
- Removal of non-physical combinations

### 2. Model Development

The following models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

### 3. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Special emphasis was placed on minimizing **false-safe predictions**, as they represent the highest safety risk.

---

## Results

### Logistic Regression

| Metric | Value |
|----------|----------|
| Accuracy | 77.42% |
| Macro F1 Score | 0.68 |

---

### XGBoost

| Metric | Value |
|----------|----------|
| Accuracy | 98.80% |
| Macro F1 Score | 0.98 |

---

### Random Forest (Best Model)

| Metric | Value |
|----------|----------|
| Accuracy | 99.15% |
| Macro F1 Score | 0.99 |
| Weighted F1 Score | 0.99 |

### Class-wise Performance

| Class | Precision | Recall | F1 Score |
|---------|----------|----------|----------|
| Marginal | 99% | 98% | 98% |
| Safe | 99% | 100% | 99% |
| Unsafe | 100% | 99% | 100% |

Random Forest achieved the highest performance and was selected as the deployment model.

---

## Features

- Real-time flight safety prediction
- Safe / Marginal / Unsafe classification
- Confidence score estimation
- Interactive Streamlit dashboard
- Feature importance analysis
- Flight envelope visualization
- AI-assisted operational safety monitoring

---

## Technology Stack

### Machine Learning
- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy

### Visualization & Deployment
- Streamlit
- Matplotlib
- Plotly

## Key Outcomes

- Developed a real-time eVTOL safety classification system.
- Achieved 99.15% classification accuracy.
- Enabled AI-driven operational safety monitoring.
- Demonstrated the feasibility of machine learning for aerospace safety assessment.
- Built a deployable proof-of-concept system for mission planning and flight envelope prediction.

---

## Future Work

- Integration with high-fidelity flight dynamics simulations.
- SHAP-based explainability analysis.
- Uncertainty-aware safety prediction.
- Hardware-in-the-loop validation.
- Integration with digital twin environments.
- Real-time avionics deployment.

---

## Impact

This project demonstrates how machine learning can be used to identify safe operating boundaries for next-generation eVTOL aircraft. By leveraging simulation-generated data, the framework enables scalable safety analysis and mission planning without relying on sensitive operational datasets.

---

## Author

**Piyusha Anand Patil**  
Machine Learning Team Lead  
Project Urdhyuth  
NMCAD Lab, Department of Aerospace Engineering  
Indian Institute of Science (IISc), Bengaluru
