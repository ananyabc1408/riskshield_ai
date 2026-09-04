from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .risk_engine import RiskEngine
from .schemas import TransactionPayload, FeedbackPayload
from .database import init_db, save_feedback, save_score, feedback_summary

BASE=Path(__file__).resolve().parent.parent
DATA=BASE/"data"/"transactions.csv"
TEST=BASE/"data"/"test_set_sample.csv"

engine=RiskEngine(DATA)
df=pd.read_csv(DATA, parse_dates=["timestamp"])
df=df.sort_values("timestamp")

init_db()

app=FastAPI(
    title="RiskShield AI",
    version="2.0",
    description="Temporal graph + behavioural ML fraud risk API"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE/"app"/"static"/"index.html")

@app.get("/app.js", include_in_schema=False)
def js():
    return FileResponse(BASE/"app"/"static"/"app.js",media_type="application/javascript")

@app.get("/styles.css", include_in_schema=False)
def css():
    return FileResponse(BASE/"app"/"static"/"styles.css",media_type="text/css")

@app.get("/api/health")
def health():
    return {"status":"ok","service":"RiskShield AI","model":"Temporal Risk Graph Ensemble"}

@app.get("/api/dashboard")
def dashboard():
    total=len(df); fraud=int(df.is_fraud.sum())
    recent=df.tail(1000)
    avg_amount=float(recent.amount.mean())
    return {
        "total_transactions":total,
        "fraud_transactions":fraud,
        "fraud_rate":round(fraud/total*100,2),
        "customers":int(df.customer_id.nunique()),
        "merchants":int(df.merchant_id.nunique()),
        "devices":int(df.device_id.nunique()),
        "avg_amount":round(avg_amount,2),
        "model_metrics":engine.stats,
        "feedback":feedback_summary()
    }

@app.get("/api/transactions")
def transactions(
    limit:int=Query(50,ge=1,le=500),
    risk_only:bool=False
):
    work=df.copy()
    if risk_only:
        work=work[work["is_fraud"]==1]
    work=work.tail(limit)
    work["timestamp"]=work["timestamp"].astype(str)
    return work.to_dict(orient="records")

@app.get("/api/transactions/{transaction_id}")
def transaction(transaction_id:str):
    row=df[df.transaction_id==transaction_id]
    if row.empty:
        raise HTTPException(404,"Transaction not found")
    return row.iloc[0].to_dict()

@app.post("/api/score")
def score(payload:TransactionPayload):
    result=engine.score(payload.model_dump())
    save_score(result)
    return result

@app.get("/api/graph/{transaction_id}")
def graph(transaction_id:str):
    row=df[df.transaction_id==transaction_id]
    if row.empty: raise HTTPException(404,"Transaction not found")
    r=row.iloc[0]
    same_device=df[df.device_id==r.device_id][
        ["transaction_id","customer_id","merchant_id","amount","timestamp","is_fraud"]
    ].tail(20)
    same_customer=df[df.customer_id==r.customer_id][
        ["transaction_id","customer_id","merchant_id","amount","timestamp","is_fraud"]
    ].tail(20)
    same_merchant=df[df.merchant_id==r.merchant_id][
        ["transaction_id","customer_id","merchant_id","amount","timestamp","is_fraud"]
    ].tail(20)
    return {
        "center":r.transaction_id,
        "device_id":r.device_id,
        "customer_id":r.customer_id,
        "merchant_id":r.merchant_id,
        "device_transactions":same_device.to_dict("records"),
        "customer_transactions":same_customer.to_dict("records"),
        "merchant_transactions":same_merchant.to_dict("records")
    }

@app.post("/api/feedback")
def feedback(payload:FeedbackPayload):
    save_feedback(payload.transaction_id,payload.label,payload.analyst,payload.note)
    return {"status":"saved","message":"Analyst feedback recorded","feedback":feedback_summary()}

@app.get("/api/metrics")
def metrics():
    reference=engine.evaluate_reference(TEST) if TEST.exists() else {}
    return {"training_validation":engine.stats,"reference_test":reference}
