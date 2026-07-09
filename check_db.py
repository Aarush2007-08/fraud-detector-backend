import os
from dotenv import load_dotenv
from supabase import create_client, Client
import httpx

load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Try to get one invoice to see columns
res = supabase.table("invoices").select("*").limit(1).execute()
print("Invoice columns:", res.data[0].keys() if res.data else "No data")

# Try executing a raw SQL query via postgres REST if possible, or RPC
try:
    # This is a hacky way to check if we have an RPC function setup to run raw SQL
    r = supabase.rpc("run_sql", {"sql": "SELECT 1"}).execute()
    print("RPC result:", r.data)
except Exception as e:
    print("RPC failed:", str(e))
