from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
import os
from dotenv import load_dotenv
from fastapi import UploadFile, File
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import shap

load_dotenv()

app = FastAPI(title="Invoice Fraud Detector API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Load ML Models
try:
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('shap_explainer.pkl', 'rb') as f:
        shap_explainer = pickle.load(f)
except Exception as e:
    print("Warning: Could not load ML models:", e)
    xgb_model = None
    shap_explainer = None

class ActionRequest(BaseModel):
    status: str # 'approved', 'held', 'escalated'

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/dashboard/stats")
def get_dashboard_stats():
    # Total Invoices
    res_total = supabase.table("invoices").select("id", count="exact").execute()
    total_count = res_total.count
    
    # Flagged Invoices (risk > 80)
    res_flagged = supabase.table("predictions").select("invoice_id", count="exact").gte("risk_score", 80).execute()
    flagged_count = res_flagged.count
    
    # Value at Risk (join predictions and invoices)
    # Supabase Python client doesn't easily sum joined columns, so we fetch flagged and sum
    res_flagged_invoices = supabase.table("predictions").select("invoice_id, invoices(amount)").gte("risk_score", 80).execute()
    value_at_risk = sum([item['invoices']['amount'] for item in res_flagged_invoices.data if item.get('invoices')])
    
    # Exact Precision: True Positives (flagged and not clean) / All Flagged
    res_tp = supabase.table("predictions").select("invoice_id", count="exact").gte("risk_score", 80).neq("fraud_type", "clean").execute()
    tp_count = res_tp.count
    
    precision = (tp_count / flagged_count * 100) if flagged_count > 0 else 100.0
    
    # Exact Accuracy: (True Positives + True Negatives) / Total
    res_tn = supabase.table("predictions").select("invoice_id", count="exact").lt("risk_score", 80).eq("fraud_type", "clean").execute()
    tn_count = res_tn.count
    accuracy = ((tp_count + tn_count) / total_count * 100) if total_count > 0 else 100.0
    
    return {
        "total_invoices": total_count,
        "flagged_invoices": flagged_count,
        "value_at_risk": round(value_at_risk, 2),
        "precision": round(precision, 1),
        "accuracy": round(accuracy, 1)
    }

@app.get("/invoices")
def get_invoices(risk_min: int = 0, limit: int = 50):
    # Fetch invoices with their predictions and vendors
    response = supabase.table("predictions") \
        .select("risk_score, fraud_type, invoice_id, invoices(id, amount, date, status, vendors(name))") \
        .gte("risk_score", risk_min) \
        .order("risk_score", desc=True) \
        .limit(limit) \
        .execute()
        
    results = []
    for row in response.data:
        inv = row.get("invoices")
        if not inv: continue
        ven = inv.get("vendors")
        results.append({
            "id": inv["id"],
            "vendor_name": ven["name"] if ven else "Unknown",
            "amount": inv["amount"],
            "date": inv["date"],
            "status": inv["status"],
            "risk_score": row["risk_score"],
            "fraud_type": row["fraud_type"]
        })
    return results

@app.get("/invoices/{id}")
def get_invoice_detail(id: str):
    # Fetch invoice, vendor, and prediction
    res_inv = supabase.table("invoices").select("*, vendors(*)").eq("id", id).single().execute()
    if not res_inv.data:
        raise HTTPException(status_code=404, detail="Invoice not found")
        
    res_pred = supabase.table("predictions").select("*").eq("invoice_id", id).single().execute()
    
    invoice_data = res_inv.data
    prediction_data = res_pred.data if res_pred.data else {}
    
    return {
        "invoice": invoice_data,
        "prediction": prediction_data
    }

@app.post("/invoices/{id}/explain")
def explain_invoice(id: str):
    if not genai_client:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
    # Fetch prediction
    res_pred = supabase.table("predictions").select("*").eq("invoice_id", id).single().execute()
    if not res_pred.data:
        raise HTTPException(status_code=404, detail="Prediction not found")
        
    prediction = res_pred.data
    
    # If explanation exists, return it
    if prediction.get("explanation"):
        return {"explanation": prediction["explanation"]}
        
    # Otherwise, generate with Gemini
    risk_score = prediction.get("risk_score")
    fraud_type = prediction.get("fraud_type")
    drivers = prediction.get("drivers")
    
    prompt = f"""
    You are an expert fraud investigator. Explain why this invoice was flagged as '{fraud_type}' with a risk score of {risk_score}%.
    The key factors (SHAP drivers from the XGBoost model) are: {drivers}.
    Keep it to one short paragraph (max 3 sentences), focusing on business risk. Be direct and clear.
    """
    
    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        explanation = response.text.strip()
        
        # Save back to supabase
        supabase.table("predictions").update({"explanation": explanation}).eq("invoice_id", id).execute()
        
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

@app.post("/invoices/{id}/action")
def take_action(id: str, action: ActionRequest):
    valid_statuses = ['approved', 'held', 'escalated']
    if action.status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")
        
    res = supabase.table("invoices").update({"status": action.status}).eq("id", id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Invoice not found")
        
    return {"message": f"Invoice {id} status updated to {action.status}"}

@app.post("/invoices/upload")
async def upload_invoice(file: UploadFile = File(...)):
    if not genai_client:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
    content = await file.read()
    
    prompt = """
    Extract the following information from this invoice. Return ONLY a valid JSON object with these exact keys:
    - vendor_name: string (the name of the company issuing the invoice)
    - date: string (YYYY-MM-DD format)
    - amount: float (the total amount due)
    - line_count: integer (the number of distinct items/lines billed)
    """
    
    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, {'mime_type': file.content_type, 'data': content}]
        )
        text = response.text.strip()
        if text.startswith('```json'): text = text[7:-3]
        elif text.startswith('```'): text = text[3:-3]
        extracted = json.loads(text.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse invoice: {str(e)}")
        
    vendor_name = extracted.get('vendor_name', 'Unknown')
    inv_date = extracted.get('date', datetime.now().strftime('%Y-%m-%d'))
    amount = float(extracted.get('amount', 0))
    line_count = int(extracted.get('line_count', 1))
    
    res_ven = supabase.table("vendors").select("*").eq("name", vendor_name).execute()
    if res_ven.data:
        vendor = res_ven.data[0]
        vendor_id = vendor['id']
    else:
        vendor_id = str(uuid.uuid4())
        vendor = {'id': vendor_id, 'name': vendor_name, 'age_days': 1, 'category': 'Unknown', 'created_at': datetime.now().isoformat()}
        supabase.table("vendors").insert(vendor).execute()
        
    res_hist = supabase.table("invoices").select("*").eq("vendor_id", vendor_id).order("date", desc=True).execute()
    history = res_hist.data
    
    freq = len(history) + 1
    if history:
        last_date = datetime.fromisoformat(history[0]['date'])
        try: curr_date = datetime.fromisoformat(inv_date)
        except: curr_date = datetime.now()
        days_since_last = max(0, (curr_date - last_date).days)
    else:
        days_since_last = 30
        
    if len(history) > 1:
        hist_amounts = [h['amount'] for h in history]
        mean = np.mean(hist_amounts)
        std = np.std(hist_amounts) if np.std(hist_amounts) > 0 else 1
        amount_z = (amount - mean) / std
    else:
        amount_z = 0.0
        
    cat_z = 0.0
    round_number = 1 if amount % 100 == 0 else 0
    try: dt_date = datetime.fromisoformat(inv_date)
    except: dt_date = datetime.now()
    weekend = 1 if dt_date.weekday() in [5, 6] else 0
    po_match = 0
    ghost = 1 if vendor['age_days'] < 30 and po_match == 0 else 0
    
    features = pd.DataFrame([{
        'amount_z_score': amount_z,
        'days_since_last': days_since_last,
        'vendor_age': vendor['age_days'],
        'frequency': freq,
        'round_number_flag': round_number,
        'weekend_flag': weekend,
        'line_count': line_count,
        'ghost_flag': ghost,
        'category_z_score': cat_z,
        'po_match_int': po_match
    }])
    
    if xgb_model and shap_explainer:
        prob = xgb_model.predict_proba(features)[0, 1]
        sv = shap_explainer.shap_values(features)[0]
        feature_names = features.columns.tolist()
        top_idx = np.argsort(np.abs(sv))[-3:][::-1]
        drivers = {feature_names[j]: float(sv[j]) for j in top_idx}
        risk_score = round(prob * 100, 2)
        fraud_type = 'clean' if risk_score < 40 else ('inflated' if amount_z > 2 else ('ghost' if ghost else 'anomaly'))
    else:
        risk_score = 0
        fraud_type = 'clean'
        drivers = {}
        
    invoice_id = str(uuid.uuid4())
    inv_record = {
        'id': invoice_id, 'vendor_id': vendor_id, 'amount': amount, 'date': inv_date,
        'status': 'pending', 'line_count': line_count, 'po_match': bool(po_match),
        'created_at': datetime.now().isoformat()
    }
    supabase.table("invoices").insert(inv_record).execute()
    
    pred_record = {
        'id': str(uuid.uuid4()), 'invoice_id': invoice_id, 'risk_score': risk_score,
        'fraud_type': fraud_type, 'drivers': drivers, 'explanation': None,
        'created_at': datetime.now().isoformat()
    }
    supabase.table("predictions").insert(pred_record).execute()
    
    return {"message": "Invoice processed", "invoice_id": invoice_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
