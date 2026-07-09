from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from google import genai
import os
from dotenv import load_dotenv

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
    
    # Precision (mock metric as per spec)
    precision = 92.0
    
    return {
        "total_invoices": total_count,
        "flagged_invoices": flagged_count,
        "value_at_risk": round(value_at_risk, 2),
        "precision": precision
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
