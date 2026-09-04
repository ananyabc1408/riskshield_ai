# test.py - RiskShield AI API Test (Port 8001)
import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_health():
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print("✅ Health check:", response.json())
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_score():
    try:
        # Complete payload with all required fields
        payload = {
            "transaction_id": "TEST_TX_001",
            "timestamp": "2026-09-04T12:00:00",
            "customer_id": "CUST_TEST",
            "merchant_id": "MERCH_TEST",
            "amount": 25000,
            "device_id": "DEV_TEST",
            "location": 0,
            "failed_attempts": 5,
            "device_changed": True,
            "location_changed": True,
            "customer_avg_amount": 1200,
            "customer_frequency": 3,
            "merchant_frequency": 150,
            "velocity_1h": 8,
            "velocity_24h": 15,
            "previous_fraud_count": 0,
            "ring_score": 0.9
        }
        print(f"📤 Sending complete payload...")
        response = requests.post(f"{BASE_URL}/api/score", json=payload)
        print(f"📥 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Score result:")
            print(f"  - Risk Score: {result.get('risk_score', 'N/A')}")
            print(f"  - Level: {result.get('risk_level', 'N/A')}")
            print(f"  - Decision: {result.get('decision', 'N/A')}")
            if result.get('reasons'):
                print(f"  - Top Reason: {result['reasons'][0]}")
            return True
        else:
            print(f"❌ Score API returned status {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Score test failed: {e}")
        return False

def test_metrics():
    try:
        response = requests.get(f"{BASE_URL}/api/metrics")
        data = response.json()
        print("✅ Metrics available")
        if data.get('training_validation'):
            auc = data['training_validation'].get('validation_auc', 'N/A')
            print(f"  - Validation AUC: {auc}")
        return True
    except Exception as e:
        print(f"❌ Metrics test failed: {e}")
        return False

def test_dashboard():
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard")
        data = response.json()
        print("✅ Dashboard loaded")
        print(f"  - Total Transactions: {data.get('total_transactions', 'N/A')}")
        print(f"  - Fraud Rate: {data.get('fraud_rate', 'N/A')}%")
        return True
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_graph():
    try:
        response = requests.get(f"{BASE_URL}/api/graph/TXN_0040968")
        data = response.json()
        print("✅ Graph view loaded")
        print(f"  - Center Transaction: {data.get('center', 'N/A')}")
        print(f"  - Device: {data.get('device_id', 'N/A')}")
        print(f"  - Customer: {data.get('customer_id', 'N/A')}")
        print(f"  - Merchant: {data.get('merchant_id', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Graph test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RiskShield AI on port 8001...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Score API", test_score),
        ("Metrics API", test_metrics),
        ("Dashboard API", test_dashboard),
        ("Graph API", test_graph)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n📌 Testing {name}...")
        if test_func():
            passed += 1
        print("-" * 50)
    
    print(f"\n✨ {passed}/{len(tests)} tests passed!")
    if passed == len(tests):
        print("🎉 All tests passed! Your API is ready for submission.")
    else:
        print("⚠️ Some tests failed. Make sure the server is running:")
        print("   uvicorn app.main:app --reload --port 8001")
