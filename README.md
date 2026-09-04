# 🛡️ RiskShield AI — Temporal Graph Fraud Defense

> **Razorpay AI Builder Internship 2026 | Track 02: AI Risk Manager**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-style, self-contained fraud-risk platform built around the supplied transaction data.

## 🎯 Why This Version is Different

Instead of treating each transaction as an isolated row, RiskShield builds a **temporal relationship graph** around customers, merchants and devices.

The final risk is a calibrated ensemble of:

1. **Supervised fraud probability** — Random Forest trained on historical transaction behaviour.
2. **Behavioural anomaly score** — Isolation Forest catches unusual transactions.
3. **Temporal graph/ring score** — detects shared-device/shared-merchant clusters, unusual velocity and cross-customer device reuse.
4. **Adaptive policy engine** — converts the risk score into ALLOW / ADDITIONAL_VERIFICATION / MANUAL_REVIEW.
5. **Human-readable explanation** — top risk drivers plus a counterfactual.
6. **Analyst feedback loop** — analysts can mark a transaction as fraud/legitimate; feedback is persisted in SQLite.

This direction is aligned with current fraud-detection research, which is moving toward real-time dynamic graphs, temporal behaviour, explainability, monitoring and human oversight.

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Validation ROC-AUC** | 98.67% |
| **Validation PR-AUC** | 94.93% |
| **Validation F1** | 90.82% |
| **Reference PR-AUC** | 79.73% |
| **Inference Time** | < 100ms |
| **Transactions Processed** | 42,168 |
| **Fraud Rate** | 1.27% |

## 🌐 Access Points

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8001 | **Dashboard** - Real-time monitoring |
| http://127.0.0.1:8001/docs | **API Docs** - Swagger UI |
| http://127.0.0.1:8001/api/health | **Health Check** |
| http://127.0.0.1:8001/api/metrics | **Model Metrics** |
| http://127.0.0.1:8001/api/dashboard | **Dashboard Data** |
| http://127.0.0.1:8001/api/transactions | **Transactions** |
| http://127.0.0.1:8001/api/graph/{id} | **Graph View** |

## 🚀 Run on Windows

Open PowerShell/CMD in this folder:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Start the server (using port 8001 to avoid conflicts)
uvicorn app.main:app --reload --port 8001