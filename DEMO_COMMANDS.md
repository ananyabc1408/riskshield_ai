# Demo

## Start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Example API request

```bash
curl -X POST http://127.0.0.1:8000/api/score ^
  -H "Content-Type: application/json" ^
  -d "{\"transaction_id\":\"TX999\",\"amount\":25000,\"failed_attempts\":5,\"device_changed\":true,\"location_changed\":true,\"velocity_1h\":8,\"velocity_24h\":15,\"customer_avg_amount\":1200,\"ring_score\":0.9}"
```

Expected behaviour: high risk and `MANUAL_REVIEW`.

## Useful pages

- Dashboard: `/`
- Swagger: `/docs`
- Health: `/api/health`
- Metrics: `/api/metrics`
