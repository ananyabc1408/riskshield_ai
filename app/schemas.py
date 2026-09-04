from pydantic import BaseModel, Field
from typing import Optional, Any

class TransactionPayload(BaseModel):
    transaction_id: str = "LIVE_TX"
    timestamp: Optional[str] = None
    customer_id: str = "CUST_NEW"
    merchant_id: str = "MERCHANT_NEW"
    amount: float = Field(1000, ge=0)
    device_id: str = "DEV_NEW"
    location: int = 0
    failed_attempts: int = Field(0, ge=0)
    device_changed: bool = False
    location_changed: bool = False
    customer_avg_amount: Optional[float] = None
    customer_frequency: Optional[int] = None
    merchant_frequency: Optional[int] = None
    velocity_1h: Optional[float] = None
    velocity_24h: Optional[float] = None
    previous_fraud_count: Optional[float] = None

class FeedbackPayload(BaseModel):
    transaction_id: str
    label: int = Field(..., ge=0, le=1)
    analyst: str = "demo_analyst"
    note: str = ""

class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: int
    risk_level: str
    fraud_probability: float
    anomaly_score: float
    ring_score: float
    combined_score: float
    decision: str
    reasons: list[str]
    counterfactual: Optional[str] = None
    graph_signals: dict[str, Any] = {}
