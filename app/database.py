import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "riskshield.db"

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            label INTEGER NOT NULL,
            analyst TEXT,
            note TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scoring_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            risk_score INTEGER,
            risk_level TEXT,
            decision TEXT,
            fraud_probability REAL,
            anomaly_score REAL,
            ring_score REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_feedback(transaction_id, label, analyst, note):
    conn=connect()
    conn.execute(
        "INSERT INTO feedback(transaction_id,label,analyst,note,created_at) VALUES(?,?,?,?,?)",
        (transaction_id,label,analyst,note,datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def save_score(result):
    conn=connect()
    conn.execute("""
        INSERT INTO scoring_events(
            transaction_id,risk_score,risk_level,decision,
            fraud_probability,anomaly_score,ring_score,created_at
        ) VALUES(?,?,?,?,?,?,?,?)
    """, (
        result["transaction_id"], result["risk_score"], result["risk_level"],
        result["decision"], result["fraud_probability"],
        result["anomaly_score"], result["ring_score"], datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def feedback_summary():
    conn=connect()
    rows=conn.execute("""
        SELECT label, COUNT(*) AS n FROM feedback GROUP BY label
    """).fetchall()
    conn.close()
    return {str(r["label"]): r["n"] for r in rows}
