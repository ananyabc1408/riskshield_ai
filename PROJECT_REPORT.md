# RiskShield AI — Project Report

## Title
**RiskShield AI: Explainable Temporal Graph Fraud Detection for Real-Time Payments**

## Problem
Traditional row-wise fraud models can miss coordinated fraud where several accounts reuse devices, merchants or other identifiers. RiskShield adds a relationship-aware layer.

## Proposed solution
The platform processes a transaction through four complementary signals:

- Supervised behavioural fraud probability
- Unsupervised anomaly score
- Temporal network/ring score
- Policy decision and explanation

The final score is:

`0.50 × fraud_probability + 0.25 × anomaly_score + 0.25 × ring_score`

Policy:
- `< 0.40` → LOW → ALLOW
- `0.40–0.69` → MEDIUM → ADDITIONAL_VERIFICATION
- `>= 0.70` → HIGH → MANUAL_REVIEW

## Novelty
The novelty is not claiming a new ML algorithm. It is the **system-level combination** of:
1. causal behavioural features,
2. temporal relationship features,
3. anomaly detection,
4. explainable risk reasons,
5. counterfactual simulation,
6. analyst feedback persistence,
7. real-time REST API and monitoring dashboard.

This makes the project suitable as an applied AI/FinTech engineering project.

## Data
The supplied `transactions.csv` contains 42,168 transactions across 2,000 customers, 60 merchants and 5,865 devices, with 534 fraud labels. The project also includes `test_set_sample.csv` as a reference evaluation dataset.

## Model
Random Forest is used for supervised classification because it is robust on mixed-scale engineered transaction features and provides useful local perturbation explanations. Isolation Forest supplies an independent novelty signal. The graph layer tracks customer/device/merchant relationships incrementally over time.

## Evaluation
Training uses an 80/20 chronological split. The generated artifact stores validation ROC-AUC, PR-AUC, precision, recall and F1. PR-AUC is especially important for imbalanced fraud data.

## Backend
FastAPI provides:
- health monitoring
- dashboard statistics
- transaction retrieval
- live scoring
- graph neighbourhood inspection
- analyst feedback
- model metrics

SQLite persists score events and analyst feedback.

## Frontend
The dashboard is a dependency-light HTML/CSS/JavaScript application served directly by FastAPI. It includes:
- KPI cards
- live scoring form
- risk verdict
- component risk bars
- model metrics
- analyst queue

## Future upgrades
For a real deployment, replace SQLite with PostgreSQL, use Redis/Kafka for streaming, use a graph database for very large networks, add authentication/RBAC, model monitoring, drift detection, threshold optimisation and a proper GNN/temporal transformer.
