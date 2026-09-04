from pathlib import Path
from app.risk_engine import RiskEngine

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"

# Force a fresh model training.
for filename in ["riskshield_models.joblib", "graph_stats.json"]:
    path = ARTIFACTS / filename
    if path.exists():
        path.unlink()
        print("Removed old artifact:", path)

engine = RiskEngine(
    BASE / "data" / "transactions.csv",
    ARTIFACTS
)

print("\nRiskShield AI training complete")
for k, v in engine.stats.items():
    print(f"{k}: {v}")

print("\nSaved artifacts to:", ARTIFACTS)
