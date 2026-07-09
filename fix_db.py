import requests
import time

BASE_URL = "https://fraud-detector-backend-kpt3.onrender.com"

print("Fetching high-risk invoices via API...")
try:
    res = requests.get(f"{BASE_URL}/invoices", params={"risk_min": 80, "limit": 1000}, timeout=30)
    invoices = res.json()
except Exception as e:
    print("Failed to fetch invoices:", str(e))
    exit(1)
    
approved_high_risk = [inv for inv in invoices if inv['status'] == 'approved']
print(f"Found {len(approved_high_risk)} approved high-risk invoices.")

if approved_high_risk:
    print("Updating to 'escalated'...")
    for i, inv in enumerate(approved_high_risk):
        print(f"[{i+1}/{len(approved_high_risk)}] Updating {inv['id']}...")
        success = False
        for attempt in range(3):
            try:
                r = requests.post(f"{BASE_URL}/invoices/{inv['id']}/action", json={"status": "escalated"}, timeout=15)
                if r.status_code == 200:
                    success = True
                    break
                else:
                    print(f"  Attempt {attempt+1} got status {r.status_code}")
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {str(e)}")
            time.sleep(2)
        if not success:
            print(f"  FAILED completely for {inv['id']}")
        time.sleep(0.5)
    print("Update complete!")
else:
    print("No invoices needed updating.")
